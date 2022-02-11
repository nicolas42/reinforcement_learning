import argparse

def get_defaults():
  
  parser = argparse.ArgumentParser()

  # === Curriculum ==============================================
  
  parser.add_argument('--cur_decay', default='exp')
  parser.add_argument('--decay_rate', default=1, type=int)
  parser.add_argument('--decay', default=0.65, type=float)
  parser.add_argument('--cur_local', default=True, action='store_false')
  parser.add_argument('--cur_len', default=1200, type=int)
  parser.add_argument('--cur_num', default=3, type=int)
  parser.add_argument('--cur', default=False, action='store_true')

  # === Terrain =================================================
  
  parser.add_argument('--stair_thing', default=True, action='store_false')
  parser.add_argument('--obstacle_type', default="None", help="flat, stairs, path, jump")
  parser.add_argument('--show_detection', default=False, action='store_true')
  parser.add_argument('--num_artifacts', default=2, type=int)
  parser.add_argument('--height_coeff', default=0.07)
  parser.add_argument('--difficulty', default=1, type=int)
  parser.add_argument('--detection_dist', default=0.9, type=float)

  # === Env ======================================================

  parser.add_argument('--more_power', default=1, type=float)
  parser.add_argument('--MASTER', default=True, action='store_true')
  parser.add_argument('--dist_off_ground', default=True, action='store_true')
  parser.add_argument('--disturbances', default=True, action='store_true')
  parser.add_argument('--record_step', default=True, action='store_true')
  parser.add_argument('--dist_inc', default=0, type=int)
  parser.add_argument('--initial_disturbance', default=100, type=int)
  parser.add_argument('--final_disturbance', default=100, type=int)
  parser.add_argument('--dist_difficulty', default=0, type=int)
  parser.add_argument('--expert', default=False, action='store_true')
  parser.add_argument('--render', default=False, action='store_true')
  parser.add_argument('--e2e', default=False, action='store_true')

  # === Perception ===============================================

  parser.add_argument('--vis', default=True, action='store_false')
  parser.add_argument('--vis_type', default="depth")
  parser.add_argument('--camera_rate', default=6, type=int)
  parser.add_argument('--display_im', default=False, action='store_true')

  # === Learning =================================================
  
  parser.add_argument('--const_std', default=False, action='store_true')
  parser.add_argument('--const_lr', default=False, action='store_true')
  parser.add_argument('--max_ts', default=5e7, type=int)
  # parser.add_argument('--max_ts', default=10e7, type=int)
  # parser.add_argument('--max_ts', default=20e7, type=int)
  # parser.add_argument('--max_ts', default=30e7, type=int)
  parser.add_argument('--lr', default=3e-4, type=float)
  parser.add_argument('--vf_lr', default=3e-4, type=float)
  parser.add_argument('--std_clip', default=False, action='store_true')
  parser.add_argument('--separate_vf', default=False, action='store_true')
  parser.add_argument('--lstm_pol', default=False, action='store_true')
  parser.add_argument('--dual_value', default=False, action='store_true')
  parser.add_argument('--dual_dqn', default=False, action='store_true')

  # === Run ======================================================
  
  parser.add_argument('--folder', default='b')
  parser.add_argument('--exp', default="test")
  parser.add_argument('--control_type', default="walk", help="stop, slow,  walk, run")
  parser.add_argument('--seed', default=42, type=int)
  parser.add_argument('--eval', default=True, action='store_false')
  parser.add_argument('--hpc', default=False, action='store_true')
  parser.add_argument('--test_pol', default=False, action='store_true')
  parser.add_argument('--eval_first', default=False, action='store_true')
  parser.add_argument('--sleep', default=0.01, type=float)

  # === DQN ======================================================
 
  parser.add_argument('--dqn', default=False, action='store_true')


  # === Miscellaneous=============================================

  parser.add_argument('--debug', default=False, action='store_true')
  parser.add_argument('--multi', default=False, action='store_true')
  parser.add_argument('--all_setup', default=False, action='store_true')
  parser.add_argument('--doa', default=False, action='store_true')
  parser.add_argument('--adv', default=False, action='store_true')
  parser.add_argument('--yu', default=False, action='store_true')
  parser.add_argument('--nicks', default=False, action='store_true')

  parser.add_argument('--rand_Kp', default=False, action='store_true')
  parser.add_argument('--early_stop', default=True, action='store_false')
  parser.add_argument('--inc', default=1, type=int)
  parser.add_argument('--terrain_first', default=True, action='store_false')
  parser.add_argument('--advantage2', default=True, action='store_false')
  parser.add_argument('--include_actions', default=False, action='store_true')
  parser.add_argument('--single_pol', default=False, action='store_true')
  parser.add_argument('--comparison', default=None)
  parser.add_argument('--use_roa', default=False, action='store_true')
  parser.add_argument('--baseline', default=False, action='store_true')
  parser.add_argument('--rand_flat', default=False, action='store_true')
  parser.add_argument('--new', default=False, action='store_true')
  parser.add_argument('--box_pen', default=False, action='store_true')
  parser.add_argument('--eval_dist', default=False, action='store_true')
  parser.add_argument('--vf_only', default=False, action='store_true')
  parser.add_argument('--speed_cur', default=False, action='store_true')
  parser.add_argument('--use_base', default=False, action='store_true')
  parser.add_argument('--display_doa', default=False, action='store_true')
  parser.add_argument('--act', default=False, action='store_true')
  parser.add_argument('--forces', default=False, action='store_true')

  # parser.add_argument('--vis', default=False, action='store_true')

  parser.add_argument('--mocap', default=False, action='store_true')
  parser.add_argument('--stage3', default=False, action='store_true')
  parser.add_argument('--dqn_cur_decay', default=False, action='store_true')
  parser.add_argument('--term', default=False, action='store_true')
  parser.add_argument('--multi_robots', default=False, action='store_true')
  parser.add_argument('--supervised', default=False, action='store_true')
  parser.add_argument('--min_eps', default=0.01, type=float)
  parser.add_argument('--eps_decay', default=3000, type=int)
  parser.add_argument('--min_decay', default=0.001, type=float)
  parser.add_argument('--just_setup', default=False, action='store_true')
  parser.add_argument('--just_dqn', default=False, action='store_true')
  parser.add_argument('--old_rew', default=False, action='store_true')
  parser.add_argument('--use_classifier', default=False, action='store_true')
  
  # ==============================================================

  # return vars(parser.parse_args())
  knowns, unknowns = parser.parse_known_args()
  return vars(knowns), unknowns
