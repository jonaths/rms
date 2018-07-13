params = {
    'max_games': 1000,
    'init_epsilon': 1,
    'end_epsilon': 0.01,
    'test_period': 0.50,
    'reps': 5,
    'num_buckets': (1, 1, 8, 3),
    'state_bounds_2': [-0.23, 0.23],
    'terminal_states': [0, 1, 2, 21, 22, 23],
    'rms':{
        'rthres': 0,
        'influence': 2
    }
}