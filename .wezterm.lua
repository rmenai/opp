local wezterm = require("wezterm")
local mux = wezterm.mux
local workspace = mux.get_active_workspace()

local function sh(cmd)
	return { "zsh", "-ic", cmd .. "; exec zsh" }
end

local function main(win, gui)
	win:spawn_tab({ args = sh("yazi") })
	win:spawn_tab({ args = sh("devenv up") })
end

for _, win in ipairs(mux.all_windows()) do
	if win:get_workspace() == workspace then
		local gui = win:gui_window()
		main(win, gui)
		break
	end
end
