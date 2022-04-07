#!/bin/bash

python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task open_drawer &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task close_drawer &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task turn_off_lightbulb &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task turn_off_led &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task place_in_drawer &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task push_into_drawer &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task lift_pink_block_drawer &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task lift_pink_block_table &
python calvin/calvin_env/calvin_env/scripts/gather_task_data.py --task rotate_pink_block_right &
