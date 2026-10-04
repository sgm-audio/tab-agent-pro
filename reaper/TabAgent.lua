--[[
  Tab Agent Pro — REAPER Integration
  AI-powered audio-to-tablature transcription inside REAPER.

  Usage:
    1. Glue the audio item you want to transcribe (right-click → Glue items)
    2. Select the glued item in the arrange view
    3. Run this script (Actions → Tab Agent)
    4. MIDI files are imported to new tracks automatically

  Requirements:
    - Python 3.10+ with Tab Agent dependencies (pip install -r requirements.txt)
    - Tab Agent Pro cloned to a known path (auto-detected or set via Settings)

  Settings configured in "Tab Agent Settings" (profile, instrument, thresholds,
  export formats, output directory) are passed through to main.py as CLI flags.

  Author: Scott Mills
  License: MIT
]]--

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

  local _, fn = reaper.GetMediaSourceFileName(src, "")
  if fn == "" or not reaper.file_exists(fn) then
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

local function quote(value)
  return '"' .. tostring(value):gsub('"', '\\"') .. '"'
end

local function build_args(settings, audio_path)
  local args = { "main.py", quote(audio_path) }

  if settings.profile ~= "" then
    table.insert(args, "--profile " .. quote(settings.profile))
  end

  if settings.instrument ~= "" then
    table.insert(args, "--instrument " .. quote(settings.instrument))
  end
  if tonumber(settings.onset) then
    table.insert(args, "--onset " .. tostring(tonumber(settings.onset)))
  end
  if tonumber(settings.frame) then
    table.insert(args, "--frame " .. tostring(tonumber(settings.frame)))
  end

  if not settings.export_midi then table.insert(args, "--no-midi") end
  if not settings.export_tab then table.insert(args, "--no-tab") end
  if not settings.export_json then table.insert(args, "--no-json") end

  if settings.output_dir ~= "" then
    table.insert(args, "--output-dir " .. quote(settings.output_dir))
  end

  return args
end

local function run_pipeline(install_path, audio_path, settings)
  local python = reaper.GetOS():match("Win") and "python" or "python3"
  local args = build_args(settings, audio_path)
  local cmd = string.format('cd %s && %s %s', quote(install_path), python, table.concat(args, " "))
  reaper.ShowConsoleMsg("Tab Agent: " .. cmd .. "\n")
  -- Lua 5.1: os.execute returns the exit code; 5.2+: (ok, "exit"|"signal", code)
  local result, kind, code = os.execute(cmd)
  if result == true or result == 0 then
    return true
  end
  if kind == "exit" and code == 0 then
    return true
  end
  return false
end

-- ============================================================================
-- IMPORT MIDI
-- ============================================================================

local function import_midi(basename, output_dir)
  local midi_files = {
    { name = "Lead Guitar",     file = basename .. "_lead_guitar.mid" },
    { name = "Rhythm Guitar L", file = basename .. "_rhythm_L.mid" },
    { name = "Rhythm Guitar R", file = basename .. "_rhythm_R.mid" },
    { name = "Bass",            file = basename .. "_bass.mid" },
  }
  local imported = 0
  for _, mf in ipairs(midi_files) do
    local path = output_dir .. "/" .. mf.file
    if reaper.file_exists(path) then
      -- Insert MIDI on its own track (InsertMedia creates a new track automatically)
      reaper.InsertMedia(path, 0)
      -- Name the last created track
      local track = reaper.GetTrack(0, reaper.GetNumTracks() - 1)
      if track then
        reaper.GetSetMediaTrackInfo_String(track, "P_NAME", basename .. " - " .. mf.name, true)
      end
      imported = imported + 1
    end
  end
  return imported
end

-- ============================================================================
-- MAIN
-- ============================================================================

reaper.Undo_BeginBlock()

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
  reaper.Undo_EndBlock("Tab Agent — Cancel", -1)
  return
end

local settings = load_settings()

local basename = audio_path:match("([^/\\]+)%.[^.]+$") or "unknown"

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

if not run_pipeline(install_path, audio_path, settings) then
  reaper.MB(
    "Transcription failed.\n\n" ..
    "Check the REAPER console for details.\n" ..
    "Common: pip install -r requirements.txt, install Demucs.",
    "Tab Agent — Error", 0
  )
  reaper.Undo_EndBlock("Tab Agent — Failed", -1)
  return
end

-- main.py writes to --output-dir when given, otherwise ./output relative to install_path
local output_dir = settings.output_dir
if output_dir == "" then
  output_dir = install_path .. "/output"
end
local imported = import_midi(basename, output_dir)

reaper.Undo_EndBlock("Tab Agent — Transcribe " .. basename, -1)

if imported > 0 then
  reaper.ShowConsoleMsg(string.format("\nDone — imported %d MIDI track(s).\n", imported))
else
  reaper.MB(
    "Transcription finished but no MIDI files found in:\n  " .. output_dir ..
      "\n\nCheck the REAPER console for details.",
    "Tab Agent — No Output", 0
  )
end
