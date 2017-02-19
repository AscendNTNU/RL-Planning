#define SIM_IMPLEMENTATION
#include "sim.h"
#include "gui.h"

#include "lib/jo_gif.cpp"

#include <stdint.h>
typedef float       r32;
typedef uint64_t    u64;
typedef uint32_t    u32;
typedef uint16_t    u16;
typedef uint8_t     u08;
typedef int32_t     s32;
typedef int16_t     s16;
typedef int8_t      s08;
#define global      static
#define persist     static

#include "SDL.h"
#include "SDL_opengl.h"
#include "SDL_assert.h"
#include <stdlib.h>

#include "lib/imgui/imgui_draw.cpp"
#include "lib/imgui/imgui.cpp"
#include "lib/imgui/imgui_demo.cpp"
#include "lib/imgui/imgui_impl_sdl.cpp"

// Allocate thirty minutes worth of real time history
#define History_Max_Length ((int)(30.0f * 60.0f / Sim_Timestep))

#define Assert SDL_assert
#define Printf SDL_Log

struct Color
{
    r32 r, g, b, a;
};

struct VideoMode
{
    SDL_Window *window;

    int width;
    int height;
    int gl_major;
    int gl_minor;
    int double_buffer;
    int depth_bits;
    int stencil_bits;
    int multisamples;

    // 0 for immediate updates, 1 for updates synchronized with the
    // vertical retrace. If the system supports it, you may
    // specify -1 to allow late swaps to happen immediately
    // instead of waiting for the next retrace.
    int swap_interval;

    // Instead of using vsync, you can specify a desired framerate
    // that the application will attempt to keep. If a frame rendered
    // too fast, it will sleep the remaining time. Leave swap_interval
    // at 0 when using this.
    int fps_lock;
};

const char *gl_error_message(GLenum error)
{
    switch (error)
    {
    case 0: return "NO_ERROR";
    case 0x0500: return "INVALID_ENUM";
    case 0x0501: return "INVALID_VALUE";
    case 0x0502: return "INVALID_OPERATION";
    case 0x0503: return "STACK_OVERFLOW";
    case 0x0504: return "STACK_UNDERFLOW";
    case 0x0505: return "OUT_OF_MEMORY";
    case 0x0506: return "INVALID_FRAMEBUFFER_OPERATION";
    default: return "UNKNOWN";
    }
}

u64 get_tick()
{
    return SDL_GetPerformanceCounter();
}

r32 get_elapsed_time(u64 begin, u64 end)
{
    u64 frequency = SDL_GetPerformanceFrequency();
    return (r32)(end - begin) / (r32)frequency;
}

r32 time_since(u64 then)
{
    u64 now = get_tick();
    return get_elapsed_time(then, now);
}

global r32 NDC_SCALE_X;
global r32 NDC_SCALE_Y;

global sim_State STATE;
global sim_State HISTORY_STATE[History_Max_Length];
global sim_Command HISTORY_CMD[History_Max_Length];
global int HISTORY_LENGTH;

void add_history(sim_Command cmd, sim_State state)
{
    if (HISTORY_LENGTH < History_Max_Length)
    {
        HISTORY_CMD[HISTORY_LENGTH] = cmd;
        HISTORY_STATE[HISTORY_LENGTH] = state;
        HISTORY_LENGTH++;
    }
}

void world_to_ndc(r32 x_world, r32 y_world,
                  r32 *x_ndc, r32 *y_ndc)
{
    *x_ndc = (x_world - 10.0f) * NDC_SCALE_X;
    *y_ndc = (y_world - 10.0f) * NDC_SCALE_Y;
}

void vertex2f(r32 x, r32 y)
{
    r32 x_ndc, y_ndc;
    world_to_ndc(x, y, &x_ndc, &y_ndc);
    glVertex2f(x_ndc, y_ndc);
}

void color4f(Color color)
{
    glColor4f(color.r, color.g, color.b, color.a);
}

void draw_line(r32 x1, r32 y1, r32 x2, r32 y2)
{
    vertex2f(x1, y1);
    vertex2f(x2, y2);
}

void fill_square(r32 x1, r32 y1, r32 x2, r32 y2)
{
    r32 x1n, y1n, x2n, y2n;
    world_to_ndc(x1, y1, &x1n, &y1n);
    world_to_ndc(x2, y2, &x2n, &y2n);
    glVertex2f(x1n, y1n);
    glVertex2f(x2n, y1n);
    glVertex2f(x2n, y2n);
    glVertex2f(x2n, y2n);
    glVertex2f(x1n, y2n);
    glVertex2f(x1n, y1n);
}

void fill_circle(r32 x, r32 y, r32 r)
{
    r32 xn, yn;
    world_to_ndc(x, y, &xn, &yn);
    u32 n = 32;
    for (u32 i = 0; i < n; i++)
    {
        r32 t1 = TWO_PI * i / (r32)n;
        r32 t2 = TWO_PI * (i + 1) / (r32)n;
        r32 x1 = x + r*cos(t1);
        r32 y1 = y + r*sin(t1);
        r32 x2 = x + r*cos(t2);
        r32 y2 = y + r*sin(t2);
        r32 x1n, y1n, x2n, y2n;
        world_to_ndc(x1, y1, &x1n, &y1n);
        world_to_ndc(x2, y2, &x2n, &y2n);
        glVertex2f(xn, yn);
        glVertex2f(x1n, y1n);
        glVertex2f(x2n, y2n);
        glVertex2f(x2n, y2n);
        glVertex2f(xn, yn);
        glVertex2f(xn, yn);
    }
}

void draw_circle(r32 x, r32 y, r32 r, u32 n = 32)
{
    for (u32 i = 0; i < n; i++)
    {
        r32 t1 = TWO_PI * i / (r32)n;
        r32 t2 = TWO_PI * (i + 1) / (r32)n;
        r32 x1 = x + r*cos(t1);
        r32 y1 = y + r*sin(t1);
        r32 x2 = x + r*cos(t2);
        r32 y2 = y + r*sin(t2);
        draw_line(x1, y1, x2, y2);
    }
}

void draw_robot(sim_Robot robot)
{
    r32 x = robot.x;
    r32 y = robot.y;
    r32 l = robot.L;
    r32 q = robot.q;
    draw_circle(x, y, 0.5f*l);
    draw_line(x, y, x + l*cos(q), y + l*sin(q));
}

struct FileData
{
    int length;
    u32 seed;
    sim_Command cmds[History_Max_Length];
};

bool write_history(char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return false;
    static FileData data;
    data.seed = STATE.seed;
    data.length = HISTORY_LENGTH;
    for (int i = 0; i < HISTORY_LENGTH; i++)
        data.cmds[i] = HISTORY_CMD[i];
    fwrite((char*)&data, 1, sizeof(data), f);
    fclose(f);
    return true;
}

bool read_history(char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        Printf("Failed to open file\n");
        return false;
    }
    static FileData data;
    fread((char*)&data, sizeof(data), 1, f);
    HISTORY_LENGTH = data.length;
    STATE = sim_init(data.seed);
    for (int i = 0; i < HISTORY_LENGTH; i++)
    {
        HISTORY_CMD[i] = data.cmds[i];
        STATE = sim_tick(STATE, HISTORY_CMD[i]);
        HISTORY_STATE[i] = STATE;
    }
    fclose(f);
    return true;
}

void gui_tick(VideoMode mode, r32 gui_time, r32 gui_dt)
{
    persist bool flag_DrawDroneGoto     = true;
    persist bool flag_DrawDrone         = true;
    persist bool flag_DrawVisibleRegion = true;
    persist bool flag_DrawTargets       = true;
    persist bool flag_DrawObstacles     = true;
    persist bool flag_Paused            = false;
    persist bool flag_Recording         = false;
    persist bool flag_SetupRecord       = false;

    persist int record_from = 0;
    persist int record_to = 0;
    persist int record_frame_skip = 1;
    persist int record_width = 0;
    persist int record_height = 0;
    persist float record_region_x = -1.0f;
    persist float record_region_y = -1.0f;
    persist float record_region_scale = 2.0f;
    persist jo_gif_t record_gif;

    persist int seek_cursor = 0;
    persist int selected_target = -1;

    persist Color color_Clear          = { 0.00f, 0.00f, 0.00f, 1.00f };
    persist Color color_Tiles          = { 0.20f, 0.35f, 0.46f, 0.66f };
    persist Color color_Grid           = { 0.00f, 0.00f, 0.00f, 1.00f };
    persist Color color_VisibleRegion  = { 0.87f, 0.93f, 0.84f, 0.50f };
    persist Color color_GreenLine      = { 0.10f, 1.00f, 0.20f, 1.00f };
    persist Color color_SelectedTarget = { 0.85f, 0.34f, 0.32f, 1.00f };
    persist Color color_Targets        = { 0.85f, 0.83f, 0.37f, 1.00f };
    persist Color color_Obstacles      = { 0.43f, 0.76f, 0.79f, 1.00f };
    persist Color color_Drone          = { 0.87f, 0.93f, 0.84f, 0.50f };
    persist Color color_DroneGoto      = { 0.87f, 0.93f, 0.84f, 0.50f };
    #define RGBA(C) C.r, C.g, C.b, C.a

    persist float send_timer = 0.0f;
    persist float send_interval = 1.0f; // In simulation time units

    NDC_SCALE_X = (mode.height / (r32)mode.width) / 12.0f;
    NDC_SCALE_Y = 1.0f / 12.0f;

    if (flag_Recording || flag_SetupRecord)
    {
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        float Ax = 2.0f / record_region_scale;
        float Bx = -1.0f - Ax*record_region_x;
        float Ay = 2.0f / record_region_scale;
        float By = -1.0f - Ay*record_region_y;
        float modelview[] = {
            Ax,   0.0f, 0.0f, 0.0f,
            0.0f, Ay,   0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            Bx,   By,   0.0f, 1.0f
        };
        glLoadMatrixf(modelview);
    }
    else
    {
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }

    if (!flag_Paused)
    {
        if (flag_Recording)
        {
            if (seek_cursor >= record_to)
            {
                flag_Paused = true;
                flag_Recording = false;
                seek_cursor = record_from;
                jo_gif_end(&record_gif);
            }
            else if (seek_cursor + record_frame_skip >= record_to)
            {
                // clamp to end
                seek_cursor = record_to;
            }
            else
            {
                seek_cursor += record_frame_skip;
            }
        }
        else if (seek_cursor < HISTORY_LENGTH-1)
        {
            seek_cursor++;
        }
        else
        {
            sim_Command cmd;

            if (!sim_recv_cmd(&cmd))
            {
                cmd.type = sim_CommandType_NoCommand;
                cmd.x = 0.0f;
                cmd.y = 0.0f;
                cmd.i = 0;
            }

            STATE = sim_tick(STATE, cmd);
            add_history(cmd, STATE);
            seek_cursor = HISTORY_LENGTH-1;

            send_timer -= Sim_Timestep;
            if (send_timer <= 0.0f)
            {
                sim_send_state(&STATE);
                send_timer += send_interval;
            }
        }
    }

    sim_State draw_state = HISTORY_STATE[seek_cursor];
    sim_Drone drone = draw_state.drone;
    sim_Robot *robots = draw_state.robots;
    sim_Robot *targets = draw_state.robots;
    sim_Robot *obstacles = draw_state.robots + Num_Targets;

    if (flag_Recording || flag_SetupRecord)
    {
        glViewport(0, 0, record_width, record_height);
    }
    else
    {
        glViewport(0, 0, mode.width, mode.height);
    }

    glClearColor(RGBA(color_Clear));
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(2.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // draw grid tiles
    glBegin(GL_TRIANGLES);
    {
        //Color color_Reward = { 0.0f, 0.0f, 0.0f, 0.9f };
        //float value = 0;
        color4f(color_Tiles);
        
        for (int yi = 0; yi < 20; yi++)
        for (int xi = 0; xi < 20; xi++)
        {
            r32 x = xi*1.0f;
            r32 y = yi*1.0f;
            //value = gridValue(x,y)/100;
            //color_Reward = {0.00f, value, 0.0f, 0.9f};
            //color4f(color_Reward);
            fill_square(x, y, x+1.0f, y+1.0f);
        }
    }
    glEnd();

    glBegin(GL_LINES);
    {
        // draw grid lines
        color4f(color_Grid);
        for (int i = 0; i <= 20; i++)
        {
            r32 x = (r32)i;
            draw_line(x, 0.0f, x, 20.0f);
            draw_line(0.0f, x, 20.0f, x);
        }

        // draw visible region
        if (flag_DrawVisibleRegion)
        {
            color4f(color_VisibleRegion);
            draw_circle(drone.x, drone.y, 2.5f);
        }

        // draw green line
        color4f(color_GreenLine);
        draw_line(0.0f, 20.0f, 20.0f, 20.0f);

        // draw targets
        if (flag_DrawTargets)
        {
            color4f(color_Targets);
            for (int i = 0; i < Num_Targets; i++)
                draw_robot(targets[i]);
            if (selected_target >= 0)
            {
                color4f(color_SelectedTarget);
                draw_robot(targets[selected_target]);
            }
        }

        // draw obstacles
        if (flag_DrawObstacles)
        {
            color4f(color_Obstacles);
            for (int i = 0; i < Num_Obstacles; i++)
                draw_robot(obstacles[i]);
        }

        // draw drone
        if (flag_DrawDrone)
        {
            color4f(color_Drone);
            draw_line(drone.x - 0.5f, drone.y,
                      drone.x + 0.5f, drone.y);
            draw_line(drone.x, drone.y - 0.5f,
                      drone.x, drone.y + 0.5f);
        }

        // draw drone goto
        if (flag_DrawDroneGoto)
        {
            color4f(color_DroneGoto);
            draw_circle(drone.xr, drone.yr, 0.45f);
        }

        // draw indicators of magnet or bumper activations
        for (int i = 0; i < Num_Targets; i++)
        {
            r32 x = targets[i].x;
            r32 y = targets[i].y;
            if (targets[i].action.was_bumped)
            {
                glColor4f(1.0f, 0.3f, 0.1f, 1.0f);
                draw_circle(x, y, 0.5f);
            }
            else if (targets[i].action.was_top_touched)
            {
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
                draw_circle(x, y, 0.5f);
            }
        }
    }
    glEnd();

    // TODO: Change capture res?
    if (flag_Recording)
    {
        static unsigned char capture_data[1024*1024*4];
        Assert(sizeof(capture_data) >=
               record_width*record_height*4);
        glReadPixels(0, 0,
                     record_width,
                     record_height,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     capture_data);

        jo_gif_frame(&record_gif, capture_data,
                     2, false);
    }

    ImGui_ImplSdl_NewFrame(mode.window);

    // DRAW FLAGS
    if (ImGui::CollapsingHeader("Rendering"))
    {
        ImGui::Checkbox("Drone goto", &flag_DrawDroneGoto);
        ImGui::Checkbox("Drone", &flag_DrawDrone);
        ImGui::Checkbox("Visible region", &flag_DrawVisibleRegion);
        ImGui::Checkbox("Targets", &flag_DrawTargets);
        ImGui::Checkbox("Obstacles", &flag_DrawObstacles);
    } // END DRAW FLAGS

    // COLORS
    if (ImGui::CollapsingHeader("Colors"))
    {
        ImGui::ColorEdit4("Clear", &color_Clear.r);
        ImGui::ColorEdit4("Grid", &color_Grid.r);
        ImGui::ColorEdit4("VisibleRegion", &color_VisibleRegion.r);
        ImGui::ColorEdit4("GreenLine", &color_GreenLine.r);
        ImGui::ColorEdit4("Targets", &color_Targets.r);
        ImGui::ColorEdit4("Obstacles", &color_Obstacles.r);
        ImGui::ColorEdit4("Drone", &color_Drone.r);
        ImGui::ColorEdit4("DroneGoto", &color_DroneGoto.r);
    } // END COLORS

    // REWIND HISTORY
    if (ImGui::CollapsingHeader("Seek##header"))
    {
        ImGui::Checkbox("Paused", &flag_Paused);
        ImGui::SliderInt("Seek##bar", &seek_cursor, 0, HISTORY_LENGTH-1);
        ImGui::InputInt("Seek frame", &seek_cursor);
        if (seek_cursor < 0) seek_cursor = 0;
        if (seek_cursor > HISTORY_LENGTH-1) seek_cursor = HISTORY_LENGTH-1;
        ImGui::Text("Time: %.2f seconds", (seek_cursor+1) * Sim_Timestep);
    } // END REWIND HISTORY

    // ROBOTS
    if (ImGui::CollapsingHeader("Robots"))
    {
        ImGui::Columns(4, "RobotsColumns");
        ImGui::Separator();
        ImGui::Text("ID"); ImGui::NextColumn();
        ImGui::Text("X"); ImGui::NextColumn();
        ImGui::Text("Y"); ImGui::NextColumn();
        ImGui::Text("Angle"); ImGui::NextColumn();
        ImGui::Separator();
        for (int i = 0; i < Num_Targets; i++)
        {
            char label[32];
            sprintf(label, "%02d", i);
            if (ImGui::Selectable(label, selected_target == i, ImGuiSelectableFlags_SpanAllColumns))
                selected_target = i;
            ImGui::NextColumn();
            ImGui::Text("%.2f", robots[i].x); ImGui::NextColumn();
            ImGui::Text("%.2f", robots[i].y); ImGui::NextColumn();
            ImGui::Text("%.2f", robots[i].q); ImGui::NextColumn();
        }
        ImGui::Columns(1);
        ImGui::Separator();
    }
    else
    {
        selected_target = -1;
    } // END ROBOTS

    // COMMUNICATION
    if (ImGui::CollapsingHeader("Communication"))
    {
        ImGui::TextWrapped("The rate at which the state is "
                           "sent can be changed using this slider. "
                           "The slider value represents the time "
                           "interval (in simulation time) "
                           "between each send.");
        ImGui::SliderFloat("Send interval", &send_interval,
                           Sim_Timestep, 1.0f);
        ImGui::Separator();

        ImGui::Text("Last 10 non-trivial commands received:");
        ImGui::Columns(5, "CommunicationColumns");
        ImGui::Separator();
        ImGui::Text("Time"); ImGui::NextColumn();
        ImGui::Text("type"); ImGui::NextColumn();
        ImGui::Text("x"); ImGui::NextColumn();
        ImGui::Text("y"); ImGui::NextColumn();
        ImGui::Text("i"); ImGui::NextColumn();
        ImGui::Separator();
        int count = 0;
        for (int i = 0; count < 10 && i <= seek_cursor; i++)
        {
            sim_State state_i = HISTORY_STATE[seek_cursor-i];
            sim_Command cmd_i = HISTORY_CMD[seek_cursor-i];
            if (cmd_i.type == sim_CommandType_NoCommand)
                continue;
            char label[32];
            sprintf(label, "%.2f", state_i.elapsed_time);
            ImGui::Selectable(label, false,
                              ImGuiSelectableFlags_SpanAllColumns);
            ImGui::NextColumn();
            switch (cmd_i.type)
            {
                case sim_CommandType_NoCommand:
                {
                    ImGui::Text("Nothing"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                } break;
                case sim_CommandType_LandInFrontOf:
                {
                    ImGui::Text("Land 180"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("%d", cmd_i.i); ImGui::NextColumn();
                } break;
                case sim_CommandType_LandOnTopOf:
                {
                    ImGui::Text("Land 45"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("%d", cmd_i.i); ImGui::NextColumn();
                } break;
                case sim_CommandType_Track:
                {
                    ImGui::Text("Track"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                    ImGui::Text("%d", cmd_i.i); ImGui::NextColumn();
                } break;
                case sim_CommandType_Search:
                {
                    ImGui::Text("Search"); ImGui::NextColumn();
                    ImGui::Text("%.2f", cmd_i.x); ImGui::NextColumn();
                    ImGui::Text("%.2f", cmd_i.y); ImGui::NextColumn();
                    ImGui::Text("-"); ImGui::NextColumn();
                } break;
            }
            count++;
        }
        ImGui::Columns(1);
        ImGui::Separator();
    } // END COMMUNICATION

    // RECORDING GIFS
    if (ImGui::CollapsingHeader("Recording"))
    {
        flag_SetupRecord = true;
        if (ImGui::Button("Mark frame as begin"))
        {
            record_from = seek_cursor;
        }
        ImGui::SameLine();
        ImGui::Text("Record from: %d", record_from);
        if (ImGui::Button("Mark frame as end"))
        {
            record_to = seek_cursor;
        }
        ImGui::SameLine();
        ImGui::Text("Record to: %d", record_to);
        ImGui::InputInt("Frame skip", &record_frame_skip);

        ImGui::InputInt("Record width", &record_width);
        ImGui::InputInt("Record height", &record_height);
        if (record_width <= 0) record_width = mode.width;
        if (record_height <= 0) record_height = mode.height;

        ImGui::SliderFloat("Record x", &record_region_x, -1.0f, 1.0f);
        ImGui::SliderFloat("Record y", &record_region_y, -1.0f, 1.0f);
        ImGui::SliderFloat("Record scale", &record_region_scale, 0.0f, 2.0f);

        if (ImGui::Button("Start recording") && !flag_Recording)
            ImGui::OpenPopup("Save recording as?");
        if (ImGui::BeginPopupModal("Save recording as?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
        {
            persist char filename[1024];
            persist bool init_filename = true;
            if (init_filename)
            {
                sprintf(filename, "recording%u.gif", STATE.seed);
                init_filename = false;
            }
            ImGui::InputText("Filename", filename, sizeof(filename));
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120,0)))
            {
                flag_Recording = true;
                flag_Paused = false;
                seek_cursor = record_from-1;
                record_gif = jo_gif_start(filename, (short)record_width, (short)record_height, 0, 32);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120,0)))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        if (ImGui::Button("Stop recording") && flag_Recording)
        {
            flag_Recording = false;
            flag_Paused = true;
            jo_gif_end(&record_gif);
        }

        if (record_from < 0) record_from = 0;
        if (record_from > HISTORY_LENGTH-1) record_from = HISTORY_LENGTH-1;
        if (record_to < record_from) record_to = record_from;
        if (record_to > HISTORY_LENGTH-1) record_to = HISTORY_LENGTH-1;
        if (record_frame_skip < 1) record_frame_skip = 1;
        ImGui::Separator();
    }
    else
    {
        flag_SetupRecord = false;
    } // END RECORDING GIFS

    // RESET AND SET SEED
    persist int custom_seed = 0;
    if (ImGui::Button("Reset"))
    {
        if (custom_seed > 0)
            STATE = sim_init((u32)custom_seed);
        else
            STATE = sim_init((u32)get_tick());
        HISTORY_LENGTH = 0;
        sim_Command cmd;
        cmd.type = sim_CommandType_NoCommand;
        add_history(cmd, STATE);
    }
    ImGui::SameLine();
    ImGui::InputInt("Seed", &custom_seed);
    // END RESET AND SET SEED

    // SAVE SIMULATION
    if (ImGui::Button("Save.."))
        ImGui::OpenPopup("Save as?");
    if (ImGui::BeginPopupModal("Save as?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        persist char filename[1024];
        persist bool init_filename = true;
        if (init_filename)
        {
            sprintf(filename, "simulation%u", STATE.seed);
            init_filename = false;
        }
        ImGui::InputText("Filename", filename, sizeof(filename));
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120,0)))
        {
            write_history(filename);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120,0)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    } // END SAVE SIMULATION

    ImGui::SameLine();

    // SAVE SINGLE SNAPSHOT
    persist bool init_snapshot_filename = true;
    if (ImGui::Button("Save snapshot.."))
        ImGui::OpenPopup("Save snapshot as?");
    if (ImGui::BeginPopupModal("Save snapshot as?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        persist char filename[1024];
        ImGui::TextWrapped("The filename is relative to the executable,"
                           "unless you write an absolute path.");
        if (init_snapshot_filename)
        {
            sprintf(filename, "snapshot%u-%u", STATE.seed, seek_cursor);
            init_snapshot_filename = false;
        }
        ImGui::InputText("Filename", filename, sizeof(filename));
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120,0)))
        {
            sim_Observed_State snapshot =
                sim_observe_state(HISTORY_STATE[seek_cursor]);
            //printf("%.2f\n", snapshot.obstacle_q[0]);
            sim_write_snapshot(filename, snapshot);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120,0)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    else
    {
        init_snapshot_filename = true;
    } // END SAVE SIMULATION

    ImGui::SameLine();

    // LOAD SIMULATION
    if (ImGui::Button("Load.."))
        ImGui::OpenPopup("Load file?");
    if (ImGui::BeginPopupModal("Load file?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
        persist char filename[1024];
        persist bool init_filename = true;
        if (init_filename)
        {
            sprintf(filename, "simulation%u", STATE.seed);
            init_filename = false;
        }
        ImGui::InputText("Filename", filename, sizeof(filename));
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120,0)))
        {
            read_history(filename);
            seek_cursor = 0;
            flag_Paused = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120,0)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    } // END LOAD SIMULATION

    ImGui::Render();
} // END gui_tick

int main(int argc, char *argv[])
{
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        Printf("Failed to initialize SDL: %s\n", SDL_GetError());
        return -1;
    }

    VideoMode mode = {};
    mode.width = 800;
    mode.height = 600;
    mode.gl_major = 1;
    mode.gl_minor = 5;
    mode.double_buffer = 1;
    mode.depth_bits = 24;
    mode.stencil_bits = 8;
    mode.multisamples = 4;
    mode.swap_interval = 0;

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, mode.gl_major);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, mode.gl_minor);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,          mode.double_buffer);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,            mode.depth_bits);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE,          mode.stencil_bits);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS,    mode.multisamples>0?1:0);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES,    mode.multisamples);

    mode.window = SDL_CreateWindow(
        "World Simulator",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        mode.width, mode.height,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    if (!mode.window)
    {
        Printf("Failed to create a window: %s\n", SDL_GetError());
        return -1;
    }

    SDL_GLContext context = SDL_GL_CreateContext(mode.window);

    // Note: This must be set on a valid context
    SDL_GL_SetSwapInterval(mode.swap_interval);

    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &mode.gl_major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &mode.gl_minor);
    SDL_GL_GetAttribute(SDL_GL_DOUBLEBUFFER,          &mode.double_buffer);
    SDL_GL_GetAttribute(SDL_GL_DEPTH_SIZE,            &mode.depth_bits);
    SDL_GL_GetAttribute(SDL_GL_STENCIL_SIZE,          &mode.stencil_bits);
    SDL_GL_GetAttribute(SDL_GL_MULTISAMPLESAMPLES,    &mode.multisamples);
    mode.swap_interval = SDL_GL_GetSwapInterval();

    sim_init_msgs(false);

    ImGui_ImplSdl_Init(mode.window);

    STATE = sim_init((u32)get_tick());
    HISTORY_LENGTH = 0;

    bool running = true;
    u64 initial_tick = get_tick();
    u64 last_frame_t = initial_tick;
    r32 elapsed_time = 0.0f;
    r32 delta_time = 1.0f / 60.0f;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSdl_ProcessEvent(&event);
            switch (event.type)
            {
                case SDL_WINDOWEVENT:
                {
                    switch (event.window.event)
                    {
                        case SDL_WINDOWEVENT_SIZE_CHANGED:
                        {
                            Printf("Window %d size changed to %dx%d\n",
                                    event.window.windowID,
                                    event.window.data1,
                                    event.window.data2);
                            mode.width = event.window.data1;
                            mode.height = event.window.data2;
                        } break;
                    }
                } break;

                case SDL_QUIT:
                {
                    running = false;
                    Printf("PROBLEM");
                } break;
            }
        }
        gui_tick(mode, elapsed_time, delta_time);
        SDL_GL_SwapWindow(mode.window);

        delta_time = time_since(last_frame_t);
        if (mode.fps_lock > 0)
        {
            r32 target_time = 1.0f / (r32)mode.fps_lock;
            r32 sleep_time = target_time - delta_time;
            if (sleep_time >= 0.01f)
                SDL_Delay((u32)(sleep_time * 1000.0f));
            delta_time = time_since(last_frame_t);
        }
        last_frame_t = get_tick();
        elapsed_time = time_since(initial_tick);

        GLenum error = glGetError();
        if (error != GL_NO_ERROR)
        {
            Printf("An error occurred: %s\n", gl_error_message(error));
            running = false;
        }
    }

    ImGui_ImplSdl_Shutdown();
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(mode.window);
    SDL_Quit();

    return 0;
}
