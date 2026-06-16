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
  local candidates = {
    reaper.GetResourcePath() .. "/Scripts/Tab-Agent",
    os.getenv("HOME") .. "/tab-agent-pro",
    os.getenv("HOME") .. "/Tab-Agent-Pro",
    os.getenv("USERPROFILE") .. "/tab-agent-pro",
  }
  for _, p in ipairs(candidates) do
    if reaper.file_exists(p .. "/main.py") then
      reaper.SetExtState("TabAgent", "install_path", p, true)
      return p
    end
  end
  return nil
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
-- RUN PIPELINE
-- ============================================================================

local function run_pipeline(install_path, audio_path)
  local python = reaper.GetOS():match("Win") and "python" or "python3"
  local cmd = string.format('cd "%s" && %s main.py "%s"', install_path, python, audio_path)
  reaper.ShowConsoleMsg("Tab Agent: " .. cmd .. "\n")
  return os.execute(cmd) == 0
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

local basename = audio_path:match("([^/\\]+)%.[^.]+$") or "unknown"

reaper.ShowConsoleMsg("\nTab Agent — Transcribing " .. basename .. "\n")
reaper.ShowConsoleMsg("  Audio:  " .. audio_path .. "\n")
reaper.ShowConsoleMsg("  Install: " .. install_path .. "\n\n")

if not run_pipeline(install_path, audio_path) then
  reaper.MB(
    "Transcription failed.\n\n" ..
    "Check the REAPER console for details.\n" ..
    "Common: pip install -r requirements.txt, install Demucs.",
    "Tab Agent — Error", 0
  )
  reaper.Undo_EndBlock("Tab Agent — Failed", -1)
  return
end

-- main.py writes output to ./output/ relative to install_path
local output_dir = install_path .. "/output"
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
