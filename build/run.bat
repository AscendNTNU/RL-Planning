@echo off
set CF=-Zi -nologo -Oi -Od -WX -W4 -wd4100 -wd4189 -wd4996 -fp:fast /MD
set LF=-subsystem:console -incremental:no -debug
cl %CF% -nologo ../example3.cpp /link %LF% -out:main.exe
main.exe
