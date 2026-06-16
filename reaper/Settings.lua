--[[
  Tab Agent Pro — Settings UI (REAPER)
  Configure install path, tuning preset, instrument, and export options.

  Author: Scott Mills
  License: MIT
]]--

-- Profiles in ordered form so menu numbers are deterministic
local PROFILES = {
  { key = "rock_standard",    name = "Rock Guitar (Standard EADGBE)",         tuning = {40,45,50,55,59,64}, strings=6, frets=24, onset=0.5, frame=0.3, suno=true,  low=false },
  { key = "rock_drop_d",      name = "Rock Guitar (Drop D)",                  tuning = {38,45,50,55,59,64}, strings=6, frets=24, onset=0.5, frame=0.3, suno=true,  low=false },
  { key = "classical",        name = "Classical / Clean Guitar",              tuning = {40,45,50,55,59,64}, strings=6, frets=19, onset=0.4, frame=0.25,suno=false, low=false },
  { key = "bass_5_string",    name = "5-String Bass (B-E-A-D-G)",            tuning = {23,28,33,38,43},    strings=5, frets=24, onset=0.55,frame=0.35,suno=true,  low=true  },
  { key = "bass_4_string",    name = "4-String Bass (E-A-D-G)",              tuning = {28,33,38,43},       strings=4, frets=24, onset=0.55,frame=0.35,suno=true,  low=true  },
  { key = "suno_aggressive",  name = "Suno AI (Aggressive Cleanup)",          tuning = {40,45,50,55,59,64}, strings=6, frets=24, onset=0.6, frame=0.4, suno=true,  low=false },
  { key = "suno_conservative",name = "Suno AI (Light Cleanup)",               tuning = {40,45,50,55,59,64}, strings=6, frets=24, onset=0.55,frame=0.35,suno=false, low=false },
  { key = "live_band",        name = "Live Band / Natural Recording",         tuning = {40,45,50,55,59,64}, strings=6, frets=24, onset=0.45,frame=0.25,suno=false, low=false },
}

-- Build ordered key list and menu text
local PROFILE_KEYS = {}
local PROFILE_MENU_LINES = {}
for i, p in ipairs(PROFILES) do
  PROFILE_KEYS[i] = p.key
  PROFILE_MENU_LINES[i] = i .. ". " .. p.name
end
local PROFILE_MENU = table.concat(PROFILE_MENU_LINES, "\n")

-- ============================================================================
-- PERSISTENT SETTINGS
-- ============================================================================

local EXSTATE_KEY = "TabAgent"

local function load_settings()
  local s = {}
  s.install_path = reaper.GetExtState(EXSTATE_KEY, "install_path")
  s.profile     = reaper.GetExtState(EXSTATE_KEY, "profile")
  s.instrument  = reaper.GetExtState(EXSTATE_KEY, "instrument")
  s.export_midi = reaper.GetExtState(EXSTATE_KEY, "export_midi")
  s.export_tab  = reaper.GetExtState(EXSTATE_KEY, "export_tab")
  s.export_json = reaper.GetExtState(EXSTATE_KEY, "export_json")
  if s.profile == "" then s.profile = "rock_standard" end
  if s.instrument == "" then s.instrument = "Guitar" end
  if s.export_midi == "" then s.export_midi = "1" end
  if s.export_tab == "" then s.export_tab = "1" end
  if s.export_json == "" then s.export_json = "1" end
  return s
end

local function save_settings(s)
  reaper.SetExtState(EXSTATE_KEY, "profile",     s.profile, true)
  reaper.SetExtState(EXSTATE_KEY, "instrument",  s.instrument, true)
  reaper.SetExtState(EXSTATE_KEY, "export_midi", tostring(s.export_midi and 1 or 0), true)
  reaper.SetExtState(EXSTATE_KEY, "export_tab",  tostring(s.export_tab and 1 or 0), true)
  reaper.SetExtState(EXSTATE_KEY, "export_json", tostring(s.export_json and 1 or 0), true)
  if s.install_path then
    reaper.SetExtState(EXSTATE_KEY, "install_path", s.install_path, true)
  end
end

-- ============================================================================
-- GUI
-- ============================================================================

local function show_gui()
  local s = load_settings()

  local ret, input = reaper.GetUserInputs(
    "Tab Agent — Settings",
    6,
    "Profile number (see below),Install path,Instrument (Guitar/Bass),Export MIDI (1=yes),Export Tab (1=yes),Export JSON (1=yes)",
    s.profile .. "," .. (s.install_path or "") .. "," .. s.instrument .. "," ..
    (s.export_midi or "1") .. "," .. (s.export_tab or "1") .. "," .. (s.export_json or "1")
  )

  if not ret then return end

  -- Show profile menu after dialog so user sees it next time
  reaper.ShowConsoleMsg("Tab Agent — Available Profiles:\n" .. PROFILE_MENU .. "\n")

  local parts = {}
  for part in input:gmatch("[^,]+") do
    parts[#parts+1] = part:match("^%s*(.-)%s*$")
  end

  local prof_idx = tonumber(parts[1])
  if prof_idx and prof_idx >= 1 and prof_idx <= #PROFILE_KEYS then
    s.profile = PROFILE_KEYS[prof_idx]
  elseif parts[1] ~= "" then
    s.profile = parts[1]
  end

  if parts[2] ~= "" then s.install_path = parts[2] end
  if parts[3] ~= "" then s.instrument = parts[3] end
  s.export_midi = parts[4] == "1" or parts[4] == "yes"
  s.export_tab  = parts[5] == "1" or parts[5] == "yes"
  s.export_json = parts[6] == "1" or parts[6] == "yes"

  save_settings(s)

  local prof_name = s.profile
  for _, p in ipairs(PROFILES) do
    if p.key == s.profile then prof_name = p.name; break end
  end

  reaper.MB(
    "Settings saved!\n" ..
    "Profile: " .. prof_name .. "\n" ..
    "Instrument: " .. s.instrument .. "\n" ..
    "Install: " .. (s.install_path or "(auto)") .. "\n" ..
    "Exports: " ..
      (s.export_midi and "MIDI " or "") ..
      (s.export_tab and "Tab " or "") ..
      (s.export_json and "JSON" or ""),
    "Tab Agent — Settings", 0
  )
end

-- ============================================================================
-- AUTO-DETECT ON FIRST RUN
-- ============================================================================

local function auto_detect()
  if reaper.GetExtState(EXSTATE_KEY, "install_path") ~= "" then return end
  local candidates = {
    reaper.GetResourcePath() .. "/Scripts/Tab-Agent",
    reaper.GetResourcePath() .. "/Scripts/Tab-Agent-Pro",
    os.getenv("HOME") .. "/tab-agent-pro",
    os.getenv("HOME") .. "/Tab-Agent-Pro",
    os.getenv("USERPROFILE") .. "/tab-agent-pro",
  }
  for _, p in ipairs(candidates) do
    if reaper.file_exists(p .. "/main.py") then
      reaper.SetExtState(EXSTATE_KEY, "install_path", p, true)
      reaper.MB("Tab Agent detected at:\n" .. p .. "\n\nRun 'Tab Agent' to transcribe.", "Tab Agent — Ready", 0)
      return
    end
  end
end

auto_detect()
show_gui()
