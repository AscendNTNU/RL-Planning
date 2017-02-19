Reinforcement learning algorithms for planning.

Currently includes:

    DQN algorithm that works
    A3C algorithm that doesn't

Includes simulator.


Building with the simulator library
-----------------------------------
Include the following lines in *one* source file

    #define SIM_IMPLEMENTATION
    #include "sim.h"

You may #include "sim.h" anywhere else if you want the declared symbols, as long as SIM_IMPLEMENTATION is only defined in one of your source files.

See example1.cpp for an example of a full application that communicates with the real-time simulation, and also uses the library internally.

Running the real-time simulator
-------------------------------
Linux

    $ cd build
    $ g++ ../gui.cpp -o sim -lGL `sdl2-config --cflags --libs`

OSX

    $ cd build
    $ g++ ../gui.cpp -o sim -framework OpenGL `sdl2-config --cflags --libs`

Windows

    > cd build
    > run_gui.bat
    Or compile manually:
    > cl -I../lib/sdl ../gui.cpp -MT /link -subsystem:console SDL2.lib SDL2main.lib opengl32.lib -out:gui.exe


To build .so file from .cpp file(Only works on linux)
--------------------------------
Linux:
	$ cc -fPIC -shared -o filename.so filename.cpp
