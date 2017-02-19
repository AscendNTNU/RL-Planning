@echo off

REM Compiler flags
REM -Od: Turns off optimizations and speeds up compilation
REM -Zi: Generates debug symbols
REM -WX: Treat warnings as errors
REM -W4: Highest level of warnings
REM -MD: Use DLL run-time library
set CF=-Zi -nologo -Od -WX -W4 -wd4100 -wd4189 -wd4996 /MD

REM Linker flags
REM -subsystem:console: Open a console
REM -debug: Create debugging information into .pdb
set LF=-subsystem:console -debug SDL2.lib SDL2main.lib opengl32.lib

cl %CF% -I../lib/sdl ../gui.cpp /link %LF% -out:gui.exe

gui.exe
