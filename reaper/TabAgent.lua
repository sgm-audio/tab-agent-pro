--[[
  Tab Agent Pro — REAPER Integration
  AI-powered audio-to-tablature transcription inside REAPER.

  Usage:
    1. Glue the audio item you want to transcribe (right-click → Glue items)
    2. Select the glued item in the arrange view
    3. Run this script (Actions → Tab Agent)
    4. MIDI files are imported to new tracks automatically

  Pipeline output is forwarded to the REAPER console.

  Requirements:
    - Python 3.11+ with the ONNX runtime dependencies (see README.md)
    - Tab Agent Pro cloned to a known path (auto-detected or set via Settings)

  Settings configured in "Tab Agent Settings" (profile, instrument, thresholds,
  export formats, output directory) are passed through to main.py as CLI flags.

  Author: Scott Mills
  License: MIT
]]--

local is_windows = reaper.GetOS():match("^Win") ~= nil

-- ============================================================================
-- PATH DETECTION
-- ============================================================================

local function find_install_path()
  local saved = reaper.GetExtState("TabAgent", "install_path")
  if saved ~= "" and reaper.file_exists(saved .. "/main.py") then
    return saved
  end
  local home = os.getenv("HOME") or ""
  local candidates = {
    reaper.GetResourcePath() .. "/Scripts/Tab-Agent",
    reaper.GetResourcePath() .. "/Scripts/Tab-Agent-Pro",
    home .. "/tab-agent-pro",
    home .. "/Tab-Agent-Pro",
    (os.getenv("USERPROFILE") or "") .. "/tab-agent-pro",
  }
  for _, p in ipairs(candidates) do
    if p ~= "" and reaper.file_exists(p .. "/main.py") then
      reaper.SetExtState("TabAgent", "install_path", p, true)
      return p
    end
  end
  return nil
end

-- ============================================================================
-- SETTINGS (written by Tab Agent Settings.lua)
-- ============================================================================

local function load_settings()
  local s = {}
  local function get(key, default)
    local v = reaper.GetExtState("TabAgent", key)
    if v == nil or v == "" then return default end
    return v
  end
  s.profile = get("profile", "")
  s.instrument = get("instrument", "")
  s.onset = get("onset", "")
  s.frame = get("frame", "")
  s.output_dir = get("output_dir", "")
  s.export_midi = get("export_midi", "1") == "1"
  s.export_tab = get("export_tab", "1") == "1"
  s.export_json = get("export_json", "1") == "1"
  return s
end

-- ============================================================================
-- GET SELECTED AUDIO
-- ============================================================================

local function get_audio_path()
  local item = reaper.GetSelectedMediaItem(0, 0)
  if not item then
    reaper.MB("Select a glued audio item in the arrange view first.", "Tab Agent", 0)
    return nil
  end

  local take = reaper.GetActiveTake(item)
  if not take then
    reaper.MB("Selected item has no active take.", "Tab Agent", 0)
    return nil
  end

  local src = reaper.GetMediaItemTake_Source(take)
  if not src then return nil end

  -- ReaScript's Lua binding returns the filename as its only result.
  local fn = reaper.GetMediaSourceFileName(src, "")
  if type(fn) ~= "string" or fn == "" or not reaper.file_exists(fn) then
    reaper.MB(
      "This item is not backed by a file on disk.\n\n" ..
      "Glue it first: right-click the item → Glue items\n" ..
      "Then select the new glued item and run Tab Agent again.",
      "Tab Agent — Glue Required", 0
    )
    return nil
  end
  return fn
end

-- ============================================================================
-- BUILD COMMAND
-- ============================================================================

local function quote(value, for_cd)
  if value:find("%c") then error("A command argument contains a control character.") end
  if not is_windows then
    return "'" .. value:gsub("'", [['"'"']]) .. "'"
  end
  -- cmd.exe expands percent variables even inside double quotes. Reject
  -- ambiguous shell characters rather than interpreting them as commands.
  if value:find('["%%!%^&|<>]') then
    error("A path or setting contains characters unsupported by the Windows command shell.")
  end
  -- The Windows Python runtime interprets backslashes before a closing quote.
  if not for_cd then value = value:gsub("(\\+)$", "%1%1") end
  return '"' .. value .. '"'
end

local function absolute_path(path)
  if is_windows then return path:match("^%a:[/\\]") ~= nil end
  return path:sub(1, 1) == "/"
end

local function resolve_output_dir(install_path, configured)
  if configured == "" then configured = "output" end
  if absolute_path(configured) then return configured end
  if is_windows and (configured:match("^[/\\]") or configured:match("^%a:")) then
    error("Use an absolute drive path (for example C:\\Tab Agent) or a relative output directory.")
  end
  return install_path .. "/" .. configured
end

local function fresh_output_dir(root)
  -- The OS gives us a unique candidate; Python atomically claims the actual
  -- output directory and aborts on collision instead of reusing old exports.
  local temp_path = os.tmpname()
  if type(temp_path) ~= "string" or temp_path == "" then
    error("Could not create a unique output directory name.")
  end
  os.remove(temp_path)
  local name = temp_path:match(is_windows and "[^/\\]+$" or "[^/]+$") or ""
  name = name:gsub("[^%w_-]", "")
  if name == "" then error("Could not create a unique output directory name.") end
  return root:gsub(is_windows and "[/\\]+$" or "/+$", "") .. "/tab-agent-" .. name
end

local function build_command(install_path, settings, audio_path, output_dir)
  if not absolute_path(install_path) or not absolute_path(audio_path) then
    error("Use absolute paths for the Tab Agent installation and audio source.")
  end
  local args = { "main.py", "--reaper-report-midi", "--fresh-output-dir", "--output-dir", quote(output_dir) }

  if settings.profile ~= "" then
    table.insert(args, "--profile " .. quote(settings.profile))
  end
  if settings.instrument ~= "" then
    table.insert(args, "--instrument " .. quote(settings.instrument))
  end
  for _, key in ipairs({ "onset", "frame" }) do
    local value = settings[key]
    if value ~= "" then
      local threshold = tonumber(value)
      if not threshold or not (threshold >= 0 and threshold <= 1) then
        error("Invalid " .. key .. " threshold. Use a number from 0 to 1 or leave it blank.")
      end
      table.insert(args, "--" .. key .. " " .. tostring(threshold))
    end
  end

  if not settings.export_midi then table.insert(args, "--no-midi") end
  if not settings.export_tab then table.insert(args, "--no-tab") end
  if not settings.export_json then table.insert(args, "--no-json") end
  table.insert(args, "--")
  table.insert(args, quote(audio_path))

  local cd = is_windows and "cd /d " or "cd "
  local python = is_windows and "python" or "python3"
  return cd .. quote(install_path, true) .. " && " .. python .. " " .. table.concat(args, " ") .. " 2>&1"
end

-- ============================================================================
-- PIPELINE RUNNER WITH STREAMED CONSOLE OUTPUT
--
-- Runs main.py via io.popen so its stdout/stderr ([tab-agent] progress lines)
-- are forwarded to the REAPER console. reaper.defer schedules one read at a
-- time; the pipe read can still block until the child writes or exits.
-- ============================================================================

local Pipeline = {}
Pipeline.__index = Pipeline

function Pipeline.new(install_path, audio_path, settings)
  local self = setmetatable({}, Pipeline)
  self.install_path = install_path
  self.audio_path = audio_path
  self.settings = settings
  self.midi_written = {}
  self.stream = nil
  self.finished = false
  self.success = false
  return self
end

function Pipeline:start()
  local ok, err = pcall(function()
    local output_root = resolve_output_dir(self.install_path, self.settings.output_dir)
    self.output_dir = fresh_output_dir(output_root)
    local cmd = build_command(self.install_path, self.settings, self.audio_path, self.output_dir)
    reaper.ShowConsoleMsg("Tab Agent: " .. cmd .. "\n")
    self.stream = io.popen(cmd, "r")
  end)
  if not ok or not self.stream then
    reaper.ShowConsoleMsg("Tab Agent: failed to start pipeline: " .. tostring(err or "unknown") .. "\n")
    self.finished = true
    self.success = false
    return false
  end
  return true
end

-- Poll one line of output. Returns "streaming" while the process runs, or
-- "done" once it exits (success boolean as second value).
function Pipeline:poll()
  if self.finished then return "done", self.success end
  if not self.stream then return "done", self.success end

  local line = self.stream:read("*l")
  if line then
    local suffix = line:match("^%[tab%-agent%-midi%] ([a-zA-Z_]+)$")
    if suffix then self.midi_written[suffix] = true end
    reaper.ShowConsoleMsg(line .. "\n")
    return "streaming", nil
  end

  -- End of stream: capture the exit code via stream:close()
  local ok, err, code = self.stream:close()
  self.stream = nil
  self.finished = true
  self.success = ok == true and (code == nil or code == 0)
  return "done", self.success
end

-- ============================================================================
-- IMPORT MIDI
-- ============================================================================

local function import_midi(basename, output_dir, midi_written)
  local midi_tracks = {
    { name = "Lead Guitar",     suffix = "lead_guitar" },
    { name = "Rhythm Guitar L", suffix = "rhythm_L" },
    { name = "Rhythm Guitar R", suffix = "rhythm_R" },
    { name = "Bass",            suffix = "bass" },
  }
  local imported = 0
  for _, mf in ipairs(midi_tracks) do
    local path = output_dir .. "/" .. basename .. "_" .. mf.suffix .. ".mid"
    if midi_written[mf.suffix] and reaper.file_exists(path) then
      local existing = {}
      for i = 0, reaper.GetNumTracks() - 1 do
        existing[reaper.GetTrackGUID(reaper.GetTrack(0, i))] = true
      end
      -- Mode 0 inserts on the selected track; mode 1 requests a new track.
      if reaper.InsertMedia(path, 1) > 0 then
        for i = 0, reaper.GetNumTracks() - 1 do
          local track = reaper.GetTrack(0, i)
          if not existing[reaper.GetTrackGUID(track)] then
            reaper.GetSetMediaTrackInfo_String(track, "P_NAME", basename .. " - " .. mf.name, true)
            imported = imported + 1
            break
          end
        end
      end
    end
  end
  return imported
end

-- ============================================================================
-- MAIN
-- ============================================================================

local install_path = find_install_path()
if not install_path then
  reaper.MB(
    "Tab Agent not found.\n\n" ..
    "Run 'Tab Agent Settings' to set the install path, " ..
    "or clone the repo to ~/tab-agent-pro.",
    "Tab Agent — Not Found", 0
  )
  return
end

local audio_path = get_audio_path()
if not audio_path then
  return
end

local settings = load_settings()

local filename = audio_path:match(is_windows and "[^/\\]+$" or "[^/]+$") or "unknown"
local stem = filename:match("^(.+)%.[^.]*$")
local basename = stem and stem:find("[^.]") and stem or filename
local project = reaper.EnumProjects(-1, "")

reaper.ShowConsoleMsg("\nTab Agent — Transcribing " .. basename .. "\n")
reaper.ShowConsoleMsg("  Audio:    " .. audio_path .. "\n")
reaper.ShowConsoleMsg("  Install:  " .. install_path .. "\n")
reaper.ShowConsoleMsg(
  string.format(
    "  Settings: profile=%s instrument=%s onset=%s frame=%s exports=%s%s%s\n\n",
    settings.profile ~= "" and settings.profile or "(default)",
    settings.instrument ~= "" and settings.instrument or "(default)",
    settings.onset ~= "" and settings.onset or "(profile)",
    settings.frame ~= "" and settings.frame or "(profile)",
    settings.export_midi and "MIDI " or "",
    settings.export_tab and "Tab " or "",
    settings.export_json and "JSON" or ""
  )
)

local pipe = Pipeline.new(install_path, audio_path, settings)
if not pipe:start() then
  reaper.MB(
    "Transcription failed to start.\n\n" ..
    "Check the REAPER console for details.\n" ..
    "Check that the Python environment from README.md is on REAPER's PATH.",
    "Tab Agent — Error", 0
  )
  return
end

-- Don't keep an undo block open while the child process is running.
local function poll_pipeline()
  local phase, success = pipe:poll()
  if phase == "streaming" then
    reaper.defer(poll_pipeline)
    return
  end

  if not success then
    reaper.MB("Transcription failed. Check the REAPER console for details.", "Tab Agent — Error", 0)
    return
  end
  if not settings.export_midi then
    reaper.ShowConsoleMsg(string.format("\nDone — exports written to %s (MIDI export disabled).\n", pipe.output_dir))
    return
  end
  if project ~= reaper.EnumProjects(-1, "") then
    reaper.MB("The active project changed. Import the exports manually from:\n" .. pipe.output_dir,
      "Tab Agent — Project Changed", 0)
    return
  end

  local imported = 0
  if next(pipe.midi_written) then
    reaper.Undo_BeginBlock()
    local ok, count = pcall(import_midi, basename, pipe.output_dir, pipe.midi_written)
    reaper.Undo_EndBlock("Tab Agent — Transcribe " .. basename, -1)
    if not ok then
      reaper.ShowConsoleMsg("Tab Agent: MIDI import error: " .. tostring(count) .. "\n")
      reaper.MB("Could not import MIDI. Check the REAPER console.", "Tab Agent — Error", 0)
      return
    end
    imported = count
  end
  if imported > 0 then
    reaper.ShowConsoleMsg(string.format("\nDone — imported %d MIDI track(s).\n", imported))
  else
    reaper.MB("No MIDI from this run could be imported from:\n" .. pipe.output_dir ..
      "\n\nCheck the REAPER console for details.", "Tab Agent — No Output", 0)
  end
end

reaper.defer(poll_pipeline)
