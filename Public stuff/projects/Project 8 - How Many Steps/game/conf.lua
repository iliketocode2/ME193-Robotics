function love.conf(t)
    t.identity = "how-many-steps"
    t.window.title = "How Many Steps"
    t.window.width = 1280
    t.window.height = 720
    t.window.resizable = true
    t.window.minwidth = 640
    t.window.minheight = 400
    t.window.msaa = 4
    t.modules.joystick = false
    t.modules.physics = false
end
