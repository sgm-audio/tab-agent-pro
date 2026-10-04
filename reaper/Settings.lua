--[[
  Tab Agent Pro — Settings UI (REAPER)
  Configure install path, tuning preset, instrument, thresholds, output
  directory and export options. The values are stored in REAPER's ExtState and
  consumed by reaper/TabAgent.lua, which passes them to main.py as CLI flags.

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

local function profile_defaults(key)
  for _, p in ipairs(PROFILES) do
    if p.key == key then return p end
  end
  return nil
end

-- ============================================================================
-- PERSISTENT SETTINGS
-- ============================================================================

local EXSTATE_KEY = "TabAgent"

local function load_settings()
  local s = {}
  local function get(key, default)
    local v = reaper.GetExtState(EXSTATE_KEY, key)
    if v == nil or v == "" then return default end
    return v
  end
  s.install_path = get("install_path", "")
  s.profile      = get("profile", "rock_standard")
  s.instrument   = get("instrument", "Guitar")
  s.onset        = get("onset", "")
  s.frame        = get("frame", "")
  s.output_dir   = get("output_dir", "")
  s.export_midi  = get("export_midi", "1")
  s.export_tab   = get("export_tab", "1")
  s.export_json  = get("export_json", "1")
  return s
end

local function save_settings(s)
  reaper.SetExtState(EXSTATE_KEY, "profile",     s.profile, true)
  reaper.SetExtState(EXSTATE_KEY, "instrument",  s.instrument, true)
  reaper.SetExtState(EXSTATE_KEY, "onset",       s.onset or "", true)
  reaper.SetExtState(EXSTATE_KEY, "frame",       s.frame or "", true)
  reaper.SetExtState(EXSTATE_KEY, "output_dir",  s.output_dir or "", true)
  reaper.SetExtState(EXSTATE_KEY, "export_midi", s.export_midi and "1" or "0", true)
  reaper.SetExtState(EXSTATE_KEY, "export_tab",  s.export_tab and "1" or "0", true)
  reaper.SetExtState(EXSTATE_KEY, "export_json", s.export_json and "1" or "0", true)
  if s.install_path and s.install_path ~= "" then
    reaper.SetExtState(EXSTATE_KEY, "install_path", s.install_path, true)
  end
end

-- ============================================================================
-- CSV HELPERS (GetUserInputs joins fields with commas; empty fields must be
-- preserved, which gmatch("[^,]+") does not do)
-- ============================================================================

local function split_csv(input)
  local parts = {}
  for part in (input .. ","):gmatch("(.-),") do
    parts[#parts + 1] = part:match("^%s*(.-)%s*$")
  end
  return parts
end

local function yes(value)
  local v = tostring(value or ""):lower()
  return v == "1" or v == "yes" or v == "true"
end

-- ============================================================================
-- GUI
-- ============================================================================

local function show_gui()
  local s = load_settings()

  local fields = "Profile number (see console after saving),Install path," ..
    "Instrument (Guitar/Bass),Onset threshold (blank=profile)," ..
    "Frame threshold (blank=profile),Output dir (blank=repo/output)," ..
    "Export MIDI (1=yes),Export Tab (1=yes),Export JSON (1=yes)"

  local defaults = table.concat({
    s.profile, s.install_path, s.instrument, s.onset, s.frame, s.output_dir,
    s.export_midi, s.export_tab, s.export_json,
  }, ",")

  local ret, input = reaper.GetUserInputs("Tab Agent — Settings", 9, fields, defaults)
  if not ret then return end

  -- Show profile menu after dialog so the user can pick a number next time
  reaper.ShowConsoleMsg("Tab Agent — Available Profiles:\n" .. PROFILE_MENU .. "\n")

  local parts = split_csv(input)
  if #parts < 9 then
    reaper.MB("Unexpected input — settings not saved.", "Tab Agent — Settings", 0)
    return
  end

  local prof_idx = tonumber(parts[1])
  if prof_idx and prof_idx >= 1 and prof_idx <= #PROFILE_KEYS then
    s.profile = PROFILE_KEYS[prof_idx]
  elseif parts[1] ~= "" then
    s.profile = parts[1]
  end

  if parts[2] ~= "" then s.install_path = parts[2] end
  if parts[3] ~= "" then s.instrument = parts[3] end

  -- Blank threshold fields mean "use the selected profile's values"; main.py
  -- resolves CLI flag > profile > built-in default.
  s.onset = parts[4]
  s.frame = parts[5]

  s.output_dir  = parts[6] or ""
  s.export_midi = yes(parts[7])
  s.export_tab  = yes(parts[8])
  s.export_json = yes(parts[9])

  save_settings(s)

  local prof_name = s.profile
  local p = profile_defaults(s.profile)
  if p then prof_name = p.name end

  reaper.MB(
    "Settings saved!\n" ..
    "Profile: " .. prof_name .. "\n" ..
    "Instrument: " .. s.instrument .. "\n" ..
    "Onset/Frame: " .. (s.onset ~= "" and s.onset or "(profile)") .. " / " ..
      (s.frame ~= "" and s.frame or "(profile)") .. "\n" ..
    "Install: " .. (s.install_path ~= "" and s.install_path or "(auto)") .. "\n" ..
    "Output: " .. (s.output_dir ~= "" and s.output_dir or "(repo/output)") .. "\n" ..
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
      reaper.SetExtState(EXSTATE_KEY, "install_path", p, true)
      reaper.MB("Tab Agent detected at:\n" .. p .. "\n\nRun 'Tab Agent' to transcribe.", "Tab Agent — Ready", 0)
      return
    end
  end
end

auto_detect()
show_gui()
