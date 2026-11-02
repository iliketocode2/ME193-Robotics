-- How Many Steps: shows the step counter's live output as a walk across a world
-- that unlocks real-distance landmarks. Listens for JSON from live.py on UDP 9000.

local socket = require("socket")

local PORT = 9000
local TIMEOUT = 2.0
local PX_PER_M = 36

local LANDMARKS = {
    { m = 10,     name = "Across the classroom", kind = "door" },
    { m = 30,     name = "Down the hallway",     kind = "door" },
    { m = 91.44,  name = "A football field",     kind = "goal" },
    { m = 200,    name = "Half a track lap",     kind = "flag" },
    { m = 400,    name = "One lap of the track", kind = "flag" },
    { m = 1000,   name = "One kilometer",        kind = "sign" },
    { m = 1609.3, name = "One mile",             kind = "sign" },
    { m = 5000,   name = "A 5K race",            kind = "flag" },
    { m = 8849,   name = "Height of Everest",    kind = "peak" },
    { m = 42195,  name = "A full marathon",      kind = "flag" },
}

local C = {
    sky1 = { 0.53, 0.76, 0.96 }, sky2 = { 0.93, 0.96, 1.0 },
    far = { 0.62, 0.72, 0.86 }, mid = { 0.42, 0.66, 0.48 }, ground = { 0.36, 0.55, 0.34 },
    path = { 0.86, 0.78, 0.62 }, ink = { 0.11, 0.12, 0.14 }, muted = { 0.42, 0.45, 0.5 },
    card = { 1, 1, 1, 0.95 }, accent = { 0.96, 0.42, 0.08 }, good = { 0.19, 0.64, 0.42 },
    bad = { 0.9, 0.28, 0.3 }, skin = { 0.98, 0.8, 0.66 }, shirt = { 0.24, 0.39, 0.87 },
}

local udp
local fonts = {}
local data = { steps = 0, cadence = 0, distance = 0, stride = 0.7, calibrated = false, status = "" }
local lastPacket = -math.huge
local shown = { distance = 0, cadence = 0 }
local lastStepId = 0
local stride = { phase = math.pi / 2, target = math.pi / 2, foot = 1 }
local footprints, dust, confetti = {}, {}, {}
local pulse = 0
local banner = nil
local demo = nil
local clock = 0

local function parse(msg)
    local out = {}
    for k, v in msg:gmatch('"([%w_]+)"%s*:%s*"([^"]*)"') do out[k] = v end
    for k, v in msg:gmatch('"([%w_]+)"%s*:%s*(-?[%d%.eE%+%-]+)') do out[k] = tonumber(v) end
    for k, v in msg:gmatch('"([%w_]+)"%s*:%s*(%a+)') do
        if v == "true" then out[k] = true elseif v == "false" then out[k] = false end
    end
    return out
end

local function resetWorld()
    shown.distance, lastStepId = 0, 0
    footprints, dust, confetti = {}, {}, {}
    stride.phase, stride.target = math.pi / 2, math.pi / 2
end

local function onStep()
    stride.target = stride.target + math.pi
    stride.foot = -stride.foot
    pulse = 1
    table.insert(footprints, { m = data.distance, foot = stride.foot, life = 1 })
    for _ = 1, 6 do
        table.insert(dust, { x = 0, y = 0, vx = -love.math.random(20, 80), vy = -love.math.random(10, 50),
                             r = love.math.random(3, 7), life = 1 })
    end
end

local function apply(msg)
    if msg.steps == nil then return end
    if msg.steps < data.steps then resetWorld() end
    local before = data.distance
    for k, v in pairs(msg) do data[k] = v end
    local id = msg.last_step or msg.steps
    while lastStepId < id do
        lastStepId = lastStepId + 1
        onStep()
    end
    for _, lm in ipairs(LANDMARKS) do
        if before < lm.m and data.distance >= lm.m then
            banner = { text = "Unlocked: " .. lm.name .. "!", life = 3.5 }
            for _ = 1, 80 do
                table.insert(confetti, { x = love.math.random() * love.graphics.getWidth(), y = -10,
                                         vx = love.math.random(-40, 40), vy = love.math.random(60, 200),
                                         c = { love.math.random(), love.math.random(), love.math.random() },
                                         a = love.math.random() * 6, life = 4 })
            end
        end
    end
end

function love.load()
    love.graphics.setBackgroundColor(C.sky2)
    fonts.huge = love.graphics.newFont(56)
    fonts.big = love.graphics.newFont(28)
    fonts.med = love.graphics.newFont(18)
    fonts.small = love.graphics.newFont(13)
    udp = socket.udp()
    udp:setsockname("127.0.0.1", PORT)
    udp:settimeout(0)
end

local function receive()
    while true do
        local msg = udp:receive()
        if not msg then break end
        lastPacket = clock
        apply(parse(msg))
    end
end

local function runDemo(dt)
    demo.next = demo.next - dt
    if demo.next <= 0 then
        demo.next = 60 / demo.cadence
        demo.steps = demo.steps + 1
        demo.times[#demo.times + 1] = clock
        while #demo.times > 0 and clock - demo.times[1] > 10 do table.remove(demo.times, 1) end
        local cad = #demo.times > 1 and 60 * (#demo.times - 1) / (demo.times[#demo.times] - demo.times[1]) or 0
        apply({ steps = demo.steps, last_step = demo.steps, distance = demo.steps * 0.7 * demo.speed,
                cadence = cad, stride = 0.7, calibrated = false, status = "demo" })
    end
    lastPacket = clock
end

function love.update(dt)
    clock = clock + dt
    receive()
    if demo then runDemo(dt) end
    shown.distance = shown.distance + (data.distance - shown.distance) * math.min(1, dt * 4)
    shown.cadence = shown.cadence + (data.cadence - shown.cadence) * math.min(1, dt * 3)
    stride.phase = stride.phase + (stride.target - stride.phase) * math.min(1, dt * 10)
    pulse = math.max(0, pulse - dt * 2.5)
    for i = #dust, 1, -1 do
        local p = dust[i]
        p.x, p.y, p.vy, p.life = p.x + p.vx * dt, p.y + p.vy * dt, p.vy + 60 * dt, p.life - dt * 1.6
        if p.life <= 0 then table.remove(dust, i) end
    end
    for i = #confetti, 1, -1 do
        local p = confetti[i]
        p.x, p.y, p.a, p.life = p.x + p.vx * dt, p.y + p.vy * dt, p.a + dt * 5, p.life - dt
        if p.life <= 0 then table.remove(confetti, i) end
    end
    for i = #footprints, 1, -1 do
        if shown.distance - footprints[i].m > 60 then table.remove(footprints, i) end
    end
    if banner then
        banner.life = banner.life - dt
        if banner.life <= 0 then banner = nil end
    end
end

local function hash(n)
    local x = math.sin(n * 127.1) * 43758.5453
    return x - math.floor(x)
end

local function drawSky(w, h)
    local steps = 24
    for i = 0, steps - 1 do
        local t = i / steps
        love.graphics.setColor(C.sky1[1] + (C.sky2[1] - C.sky1[1]) * t, C.sky1[2] + (C.sky2[2] - C.sky1[2]) * t,
                               C.sky1[3] + (C.sky2[3] - C.sky1[3]) * t)
        love.graphics.rectangle("fill", 0, h * t * 0.75, w, h * 0.75 / steps + 1)
    end
end

local function drawLayer(cam, factor, baseY, amp, color, period)
    love.graphics.setColor(color)
    local pts = {}
    local off = cam * factor
    for x = 0, love.graphics.getWidth() + 20, 20 do
        local wx = (x + off) / period
        pts[#pts + 1] = x
        pts[#pts + 1] = baseY - amp * (0.6 * math.sin(wx) + 0.3 * math.sin(wx * 2.3 + 1) + 0.1 * math.sin(wx * 5.1))
    end
    pts[#pts + 1] = love.graphics.getWidth() + 20
    pts[#pts + 1] = love.graphics.getHeight()
    pts[#pts + 1] = 0
    pts[#pts + 1] = love.graphics.getHeight()
    local tris = love.math.triangulate(pts)
    for _, t in ipairs(tris) do love.graphics.polygon("fill", t) end
end

local function drawLandmark(lm, x, groundY)
    love.graphics.setLineWidth(4)
    if lm.kind == "goal" then
        love.graphics.setColor(0.98, 0.85, 0.15)
        love.graphics.line(x, groundY, x, groundY - 90)
        love.graphics.line(x - 40, groundY - 90, x + 40, groundY - 90)
        love.graphics.line(x - 40, groundY - 90, x - 40, groundY - 170)
        love.graphics.line(x + 40, groundY - 90, x + 40, groundY - 170)
    elseif lm.kind == "flag" then
        love.graphics.setColor(C.ink)
        love.graphics.line(x, groundY, x, groundY - 150)
        for r = 0, 3 do
            for c = 0, 5 do
                love.graphics.setColor(((r + c) % 2 == 0) and { 0.1, 0.1, 0.1 } or { 1, 1, 1 })
                love.graphics.rectangle("fill", x + c * 10, groundY - 150 + r * 10, 10, 10)
            end
        end
    elseif lm.kind == "peak" then
        love.graphics.setColor(0.55, 0.57, 0.62)
        love.graphics.polygon("fill", x - 130, groundY, x, groundY - 230, x + 130, groundY)
        love.graphics.setColor(1, 1, 1)
        love.graphics.polygon("fill", x - 40, groundY - 160, x, groundY - 230, x + 40, groundY - 160)
    elseif lm.kind == "door" then
        love.graphics.setColor(0.55, 0.36, 0.22)
        love.graphics.rectangle("fill", x - 25, groundY - 110, 50, 110, 4)
        love.graphics.setColor(0.95, 0.8, 0.3)
        love.graphics.circle("fill", x + 14, groundY - 55, 4)
    else
        love.graphics.setColor(0.45, 0.47, 0.5)
        love.graphics.rectangle("fill", x - 30, groundY - 70, 60, 70, 18, 18)
    end
    love.graphics.setColor(C.card)
    local label = lm.name .. "  ·  " .. (lm.m >= 1000 and string.format("%.2f km", lm.m / 1000)
                                                     or string.format("%d m", math.floor(lm.m + 0.5)))
    local tw = fonts.small:getWidth(label) + 16
    love.graphics.rectangle("fill", x - tw / 2, groundY - 260, tw, 26, 6)
    love.graphics.setColor(C.ink)
    love.graphics.setFont(fonts.small)
    love.graphics.print(label, x - tw / 2 + 8, groundY - 255)
    love.graphics.setLineWidth(1)
end

local function drawWalker(x, groundY)
    local s = math.sin(stride.phase)
    local legL, legR = 0.5 * s, -0.5 * s
    local bob = -math.abs(math.cos(stride.phase)) * 4
    local hipY = groundY - 48 + bob
    love.graphics.setLineWidth(7)
    love.graphics.setColor(0.2, 0.22, 0.3)
    for _, a in ipairs({ legL, legR }) do
        local kx, ky = x + math.sin(a) * 24, hipY + math.cos(a) * 24
        love.graphics.line(x, hipY, kx, ky, kx + math.sin(a * 0.4) * 24, ky + 22)
    end
    love.graphics.setColor(C.shirt)
    love.graphics.line(x, hipY, x + 3, hipY - 42)
    love.graphics.setColor(C.skin)
    love.graphics.line(x + 2, hipY - 34, x + 2 - legL * 30, hipY - 12)
    love.graphics.line(x + 2, hipY - 34, x + 2 - legR * 30, hipY - 12)
    love.graphics.circle("fill", x + 4, hipY - 58, 13)
    love.graphics.setColor(C.accent)
    love.graphics.rectangle("fill", x - 4 + legR * 28, groundY - 6 + bob * 0.2, 16, 6, 2)
    love.graphics.setLineWidth(1)
end

local function drawWorld(w, h)
    local groundY = h * 0.78
    local cam = shown.distance * PX_PER_M
    local px = w * 0.3
    drawSky(w, h)
    love.graphics.setColor(1, 1, 1, 0.8)
    for i = 0, 5 do
        local cx = (i * 300 - cam * 0.05) % (w + 300) - 150
        love.graphics.ellipse("fill", cx, 70 + hash(i) * 80, 60, 18)
        love.graphics.ellipse("fill", cx + 35, 62 + hash(i) * 80, 40, 16)
    end
    drawLayer(cam, 0.15, groundY - 120, 70, C.far, 260)
    drawLayer(cam, 0.4, groundY - 40, 40, C.mid, 160)

    love.graphics.setColor(C.ground)
    love.graphics.rectangle("fill", 0, groundY, w, h - groundY)
    love.graphics.setColor(C.path)
    love.graphics.rectangle("fill", 0, groundY, w, 26)

    local m0 = math.floor((cam - px) / PX_PER_M) - 1
    local m1 = math.ceil((cam + w - px) / PX_PER_M) + 1
    love.graphics.setFont(fonts.small)
    for m = m0, m1 do
        if m >= 0 then
            local x = px + m * PX_PER_M - cam
            if m % 10 == 0 then
                love.graphics.setColor(C.ink)
                love.graphics.rectangle("fill", x - 1, groundY, 2, 26)
                love.graphics.print(m .. " m", x + 4, groundY + 30)
            else
                love.graphics.setColor(0.6, 0.52, 0.4)
                love.graphics.rectangle("fill", x - 1, groundY + 18, 2, 8)
            end
            if m % 7 == 3 and hash(m) > 0.4 then
                love.graphics.setColor(0.45, 0.3, 0.2)
                love.graphics.rectangle("fill", x - 4, groundY - 60, 8, 60)
                love.graphics.setColor(0.2 + hash(m + 1) * 0.15, 0.5 + hash(m + 2) * 0.2, 0.28)
                love.graphics.circle("fill", x, groundY - 75, 26 + hash(m + 3) * 12)
            end
        end
    end
    for _, lm in ipairs(LANDMARKS) do
        local x = px + lm.m * PX_PER_M - cam
        if x > -200 and x < w + 200 then drawLandmark(lm, x, groundY) end
    end
    for _, f in ipairs(footprints) do
        local x = px + f.m * PX_PER_M - cam
        local a = math.max(0.15, 1 - (shown.distance - f.m) / 40)
        love.graphics.setColor(0.45, 0.35, 0.25, a)
        love.graphics.ellipse("fill", x, groundY + 13 + f.foot * 5, 8, 4)
    end
    for _, p in ipairs(dust) do
        love.graphics.setColor(0.85, 0.78, 0.65, p.life * 0.8)
        love.graphics.circle("fill", px + p.x, groundY - 2 + p.y, p.r)
    end
    drawWalker(px, groundY)
end

local function card(x, y, w, h)
    love.graphics.setColor(C.card)
    love.graphics.rectangle("fill", x, y, w, h, 12)
end

local function drawGauge(cx, cy, r, value, maxv)
    love.graphics.setLineWidth(8)
    love.graphics.setColor(0.88, 0.89, 0.91)
    love.graphics.arc("line", "open", cx, cy, r, math.pi, 2 * math.pi, 40)
    local f = math.min(value / maxv, 1)
    local col = value < 70 and C.muted or (value < 130 and C.good or C.accent)
    love.graphics.setColor(col)
    if f > 0 then love.graphics.arc("line", "open", cx, cy, r, math.pi, math.pi + math.pi * f, 40) end
    love.graphics.setLineWidth(3)
    love.graphics.setColor(C.ink)
    local a = math.pi + math.pi * f
    love.graphics.line(cx, cy, cx + math.cos(a) * (r - 14), cy + math.sin(a) * (r - 14))
    love.graphics.circle("fill", cx, cy, 5)
    love.graphics.setLineWidth(1)
end

local function nextLandmark()
    for _, lm in ipairs(LANDMARKS) do
        if data.distance < lm.m then return lm end
    end
end

local function fmtDistance(m)
    if m >= 1000 then return string.format("%.2f km", m / 1000) end
    return string.format("%.1f m", m)
end

local function drawHud(w)
    card(20, 20, 250, 130)
    love.graphics.setColor(C.muted)
    love.graphics.setFont(fonts.small)
    love.graphics.print("STEPS", 36, 30)
    love.graphics.setFont(fonts.huge)
    love.graphics.setColor(C.ink[1] + pulse * 0.8, C.ink[2] + pulse * 0.3, C.ink[3])
    local s = 1 + pulse * 0.12
    love.graphics.print(tostring(data.steps), 34, 48, 0, s, s)

    card(290, 20, 250, 130)
    love.graphics.setColor(C.muted)
    love.graphics.setFont(fonts.small)
    love.graphics.print("DISTANCE", 306, 30)
    love.graphics.setColor(C.ink)
    love.graphics.setFont(fonts.big)
    love.graphics.print(fmtDistance(shown.distance), 306, 54)
    love.graphics.setFont(fonts.small)
    love.graphics.setColor(C.muted)
    love.graphics.print(string.format("%.2f m per step%s", data.stride or 0, data.calibrated and "" or "  (uncalibrated)"),
                        306, 96)

    card(560, 20, 230, 130)
    love.graphics.setColor(C.muted)
    love.graphics.print("CADENCE", 576, 30)
    love.graphics.setColor(C.ink)
    love.graphics.setFont(fonts.big)
    love.graphics.print(string.format("%d", math.floor(shown.cadence + 0.5)), 576, 54)
    love.graphics.setFont(fonts.small)
    love.graphics.setColor(C.muted)
    love.graphics.print("steps / min", 576, 96)
    drawGauge(722, 118, 48, shown.cadence, 200)

    local lm = nextLandmark()
    if lm and w > 1060 then
        card(810, 20, w - 830, 130)
        love.graphics.setFont(fonts.small)
        love.graphics.setColor(C.muted)
        love.graphics.print("NEXT LANDMARK", 826, 30)
        love.graphics.setColor(C.ink)
        love.graphics.setFont(fonts.med)
        love.graphics.print(lm.name, 826, 52)
        local prev = 0
        for _, l in ipairs(LANDMARKS) do if l.m < lm.m then prev = l.m end end
        local f = (data.distance - prev) / (lm.m - prev)
        love.graphics.setColor(0.88, 0.89, 0.91)
        love.graphics.rectangle("fill", 826, 90, w - 862, 12, 6)
        love.graphics.setColor(C.good)
        love.graphics.rectangle("fill", 826, 90, (w - 862) * math.max(0, math.min(1, f)), 12, 6)
        love.graphics.setColor(C.muted)
        love.graphics.setFont(fonts.small)
        love.graphics.print(fmtDistance(math.max(0, lm.m - data.distance)) .. " to go", 826, 110)
    end
    love.graphics.setFont(fonts.small)
    love.graphics.setColor(1, 1, 1, 0.85)
    love.graphics.print("sensor: " .. tostring(data.status or ""), 20, love.graphics.getHeight() - 24)
end

local function drawWaiting(w, h)
    love.graphics.setColor(0, 0, 0, 0.45)
    love.graphics.rectangle("fill", 0, 0, w, h)
    card(w / 2 - 260, h / 2 - 110, 520, 220)
    love.graphics.setColor(C.ink)
    love.graphics.setFont(fonts.big)
    local dots = string.rep(".", math.floor(clock * 2) % 4)
    local title = "Waiting for sensor" .. dots
    love.graphics.print(title, w / 2 - fonts.big:getWidth("Waiting for sensor...") / 2, h / 2 - 90)
    love.graphics.setFont(fonts.med)
    love.graphics.setColor(C.muted)
    local lines = {
        "Start the step counter:  python live.py",
        "(or replay a walk:  python live.py --replay <session>)",
        "Listening on UDP port " .. PORT,
        "Press D for a demo walk without the sensor",
    }
    for i, l in ipairs(lines) do
        love.graphics.print(l, w / 2 - fonts.med:getWidth(l) / 2, h / 2 - 40 + (i - 1) * 30)
    end
    local r = 8 + 4 * math.sin(clock * 4)
    love.graphics.setColor(C.accent)
    love.graphics.circle("fill", w / 2, h / 2 + 90, r)
end

function love.draw()
    local w, h = love.graphics.getDimensions()
    drawWorld(w, h)
    drawHud(w)
    for _, p in ipairs(confetti) do
        love.graphics.setColor(p.c[1], p.c[2], p.c[3], math.min(1, p.life))
        love.graphics.push()
        love.graphics.translate(p.x, p.y)
        love.graphics.rotate(p.a)
        love.graphics.rectangle("fill", -4, -2, 8, 4)
        love.graphics.pop()
    end
    if banner then
        local a = math.min(1, banner.life)
        love.graphics.setFont(fonts.big)
        local tw = fonts.big:getWidth(banner.text) + 40
        love.graphics.setColor(C.accent[1], C.accent[2], C.accent[3], a)
        love.graphics.rectangle("fill", w / 2 - tw / 2, h * 0.32, tw, 56, 14)
        love.graphics.setColor(1, 1, 1, a)
        love.graphics.print(banner.text, w / 2 - tw / 2 + 20, h * 0.32 + 10)
    end
    if clock - lastPacket > TIMEOUT then drawWaiting(w, h) end
end

function love.keypressed(key)
    if key == "escape" then
        love.event.quit()
    elseif key == "d" then
        if demo then
            demo = nil
        else
            resetWorld()
            data.steps, data.distance = 0, 0
            demo = { steps = 0, next = 0.5, cadence = 110, speed = 1, times = {} }
        end
    elseif key == "f" and demo then
        demo.speed = 25
    elseif key == "r" then
        resetWorld()
        data.steps, data.distance, data.cadence = 0, 0, 0
        demo = demo and { steps = 0, next = 0.5, cadence = 110, speed = 1, times = {} } or nil
    end
end

function love.quit()
    if udp then udp:close() end
end
