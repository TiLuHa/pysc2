#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

import importlib
import threading

from absl import app
from absl import flags

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
from pysc2.lib import viewer as viewer_lib


FLAGS = flags.FLAGS
flags.DEFINE_enum("render_mode", "pygame", ["none", "pygame", "web"],
                  "Viewer backend to use.")
flags.DEFINE_string("web_host", "0.0.0.0", "Web viewer host to bind.")
flags.DEFINE_integer("web_port", 8000, "Web viewer port to bind.")
flags.DEFINE_integer("web_fps", 15, "Web viewer target FPS.")
flags.DEFINE_enum("web_mode", "feature", ["feature", "rgb", "auto"],
                  "Which data to stream: feature layers, RGB render, or auto.")
flags.DEFINE_string("web_feature_screen", "composite",
                    "Feature layer to render for the screen (feature mode).")
flags.DEFINE_string("web_feature_minimap", "composite",
                    "Feature layer to render for the minimap (feature mode).")
flags.DEFINE_bool("web_feature_composite_player_relative", True,
                  "Overlay player_relative on composite feature renders.")
flags.DEFINE_bool("web_overlay_effects", True,
                  "Draw active effects on the web screen.")
flags.DEFINE_bool("web_overlay_units", True,
                  "Draw circles (and labels) for units on the web screen.")
flags.DEFINE_bool("web_overlay_labels", True,
                  "Draw unit names when web_overlay_units is enabled.")
flags.DEFINE_bool("web_overlay_camera", True,
                  "Draw camera rectangle on the minimap in the web viewer.")
flags.DEFINE_integer("web_feature_screen_size", 128,
                     "Feature screen size for web viewer (square).")
flags.DEFINE_integer("web_feature_minimap_size", 128,
                     "Feature minimap size for web viewer (square).")
point_flag.DEFINE_point("web_screen_px", None,
                        "Target pixel size for web screen canvas (w,h).")
point_flag.DEFINE_point("web_minimap_px", None,
                        "Target pixel size for web minimap canvas (w,h).")
flags.DEFINE_float("feature_camera_width", None,
                   "Camera width in world units for feature layers. "
                   "Bigger values zoom out.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", False,
                  "Whether to include raw units.")
flags.DEFINE_bool("use_camera_position", False,
                  "Whether to include the camera position in observations.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")


def _flag_present(name):
  try:
    return FLAGS[name].present
  except Exception:
    return False


def build_viewer_options():
  if FLAGS.render_mode == "none":
    return None

  if FLAGS.render_mode == "web":
    want_rgb = False
    if FLAGS.web_mode == "rgb":
      want_rgb = True
    elif FLAGS.web_mode == "auto":
      want_rgb = bool(FLAGS.rgb_screen_size or FLAGS.rgb_minimap_size)

    if want_rgb:
      if FLAGS.rgb_screen_size is None:
        FLAGS.rgb_screen_size = point.Point(256, 192)
      if FLAGS.rgb_minimap_size is None:
        FLAGS.rgb_minimap_size = point.Point(128, 128)
      if FLAGS.action_space is None:
        FLAGS.action_space = "RGB"

    if FLAGS.web_feature_screen_size and not _flag_present("feature_screen_size"):
      FLAGS.feature_screen_size = point.Point(
          FLAGS.web_feature_screen_size, FLAGS.web_feature_screen_size)
    if FLAGS.web_feature_minimap_size and not _flag_present("feature_minimap_size"):
      FLAGS.feature_minimap_size = point.Point(
          FLAGS.web_feature_minimap_size, FLAGS.web_feature_minimap_size)

    return viewer_lib.ViewerOptions(
        mode="web",
        web=viewer_lib.WebViewerOptions(
            host=FLAGS.web_host,
            port=FLAGS.web_port,
            fps=FLAGS.web_fps,
            mode=FLAGS.web_mode,
            feature_screen=FLAGS.web_feature_screen,
            feature_minimap=FLAGS.web_feature_minimap,
            composite_player_relative=FLAGS.web_feature_composite_player_relative,
            overlay_effects=FLAGS.web_overlay_effects,
            overlay_units=FLAGS.web_overlay_units,
            overlay_labels=FLAGS.web_overlay_labels,
            overlay_camera=FLAGS.web_overlay_camera,
            screen_px=FLAGS.web_screen_px,
            minimap_px=FLAGS.web_minimap_px))

  return viewer_lib.ViewerOptions(mode="pygame")


def run_thread(agent_classes, players, map_name, viewer_options=None):
  """Run one thread worth of the environment with agents."""
  with sc2_env.SC2Env(
      map_name=map_name,
      battle_net_map=FLAGS.battle_net_map,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_screen_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space,
          camera_width_world_units=FLAGS.feature_camera_width,
          use_feature_units=FLAGS.use_feature_units,
          use_raw_units=FLAGS.use_raw_units,
          use_camera_position=FLAGS.use_camera_position),
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      disable_fog=FLAGS.disable_fog,
      viewer_options=viewer_options) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agents = [agent_cls() for agent_cls in agent_classes]
    run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
    if FLAGS.save_replay:
      env.save_replay(agent_classes[0].__name__)


def main(unused_argv):
  """Run an agent."""
  if FLAGS.trace:
    stopwatch.sw.trace()
  elif FLAGS.profile:
    stopwatch.sw.enable()

  map_inst = maps.get(FLAGS.map)

  agent_classes = []
  players = []

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)
  agent_classes.append(agent_cls)
  players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                               FLAGS.agent_name or agent_name))

  if map_inst.players >= 2:
    if FLAGS.agent2 == "Bot":
      players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                 sc2_env.Difficulty[FLAGS.difficulty],
                                 sc2_env.BotBuild[FLAGS.bot_build]))
    else:
      agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
      agent_cls = getattr(importlib.import_module(agent_module), agent_name)
      agent_classes.append(agent_cls)
      players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race],
                                   FLAGS.agent2_name or agent_name))

  threads = []
  for _ in range(FLAGS.parallel - 1):
    t = threading.Thread(target=run_thread,
                         args=(agent_classes, players, FLAGS.map, None))
    threads.append(t)
    t.start()

  run_thread(agent_classes, players, FLAGS.map, build_viewer_options())

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)
