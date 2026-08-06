---
layout: post
title: I Turned Android Into a City
---

Every explanation of Android's internals I've ever read is a stack of boxes.
Zygote in one box, system_server in another, arrows between them, and you
nod along until someone asks what actually happens when you tap an icon.
Then the boxes fall apart, because the interesting part was never the boxes.
It was the traffic between them.

So I built the traffic. It's called DroidCity, and it's live at
**[thuat.dev/droidcity](https://thuat.dev/droidcity/)**. The idea isn't
mine originally, I saw [PGSimCity](https://nikolays.github.io/PGSimCity/)
render Postgres as a living city and couldn't stop thinking about what
Android would look like on the same board.

[![DroidCity, the whole board laid out as the Android stack, with a working phone screen at the front edge](/assets/droidcity-overview.jpg)](https://thuat.dev/droidcity/)

It's a 3D city where the board itself is the Android stack, read north to
south. The hardware strip sits at the top: CPU cores, the RAM bank, the disk,
a radio mast. Below that, the boot row, bootloader to kernel to init to
system_server. Then the core processes: Zygote as a foundry that stamps out
new processes, system_server as the city hall next door, SurfaceFlinger as
the last stop every frame makes before reaching the glass. Every running app
rises as its own walled ward, and at the front edge there's a phone screen
that actually works.

That last part is the piece I care about most. The screen isn't a picture of
a launcher, it is the launcher. Tap an icon and the launch travels for real:
the touch runs north to InputDispatcher inside system_server, the launcher
files an Intent, ActivityManager approves it, Zygote forks, a new ward rises
out of the ground, and its first frame lands back on the glass where you
tapped. Press Back and the back stack actually pops. Open Recents and you get
one card per live process.

Inside each ward the metaphor keeps going. The main road is the Looper, with
the ANR watchdog riding it. The Activity tower rises through its lifecycle
floors with a ViewModel sitting on the roof. There's a heap yard where GC
sweeps pass through, a render bench racing each frame to SurfaceFlinger, and
a native workshop on the JNI bridge whose heap no garbage collector will ever
touch. Hover anything and the tooltip tells you what the real Android
mechanism is.

A few doors worth walking through first:

- [A process is born](https://thuat.dev/droidcity/?story=2), the full
  tap-to-first-frame launch as a narrated chapter
- [The metal](https://thuat.dev/droidcity/?view=hardware), where PSI
  pressure builds, kswapd compresses cold pages into zram, and lmkd finally
  picks a victim off the oom_adj ladder
- [SurfaceFlinger](https://thuat.dev/droidcity/?view=surfaceflinger),
  because most explanations of Android stop right before the part where
  pixels happen
- [Play all](https://thuat.dev/droidcity/?story=all), seven chapters in slow
  motion, boot to sleep

My favorite thing to show people is the memory ladder. Open any ward, press
Allocate 150MB a few times, and watch what Android actually does under
pressure. It doesn't jump to a kill. Reclaim runs first, zram fills with
diminishing returns, and only when reclaim can't keep up does lmkd walk the
kill order, which is displayed live in the corner the whole time. Watching
cached wards get demolished in oom_adj order teaches more than any
documentation page I've found on the subject.

I'll be honest about what this is: an early prototype that simplifies
aggressively. The city takes liberties with timing, collapses some flows,
and certainly contains mistakes. I keep an audit of what's modeled
faithfully, what's simplified on purpose, and what's missing entirely in
[the reference doc](https://github.com/nthuat/droidcity/blob/main/docs/android-flow-reference.md).
If you know Android internals and spot a tooltip or a flow that's wrong,
[open an issue](https://github.com/nthuat/droidcity/issues) and tell me what
the real behavior is. Corrections are the most useful contribution this
project can get.

Start with the [full tour](https://thuat.dev/droidcity/?story=all). It runs
in slow motion so the city keeps pace with the narration, and by chapter
three you'll know why I couldn't leave this as a stack of boxes.
