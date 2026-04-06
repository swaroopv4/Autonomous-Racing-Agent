import os

class Config:
    # Environment Configuration 
    ENV_NAME = "donkey-generated-track-v0"
    
    DONKEY_SIM_PATH = "/Users/sachi/Downloads/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
    
    DONKEY_PORT = 9091
    DONKEY_CONF = {
        "exe_path": DONKEY_SIM_PATH,
        "port": DONKEY_PORT,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "team 43",
        "font_size": 20,
        "racer_name": "team 43",
        "country": "USA",
        "bio": "Learning to race with PPO!",
        "guid": "ppo-racing-agent-v1",
        "max_cte": 8.0, 
    }
    
    # Image dimensions
    IMG_HEIGHT = 120
    IMG_WIDTH = 160
    IMG_CHANNELS = 3
    
    # Sensor dimensions
    SENSOR_DIM = 4  # speed, cte, angle, progress
    
    # Action space
    ACTION_DIM = 2  # steering, throttle
    
    # Training Configuration 
    # PPO Hyperparameters - OPTIMIZED FOR TURNS
    LEARNING_RATE = 2e-4  
    GAMMA = 0.995  
    GAE_LAMBDA = 0.97  
    CLIP_EPSILON = 0.2
    EPOCHS_PER_UPDATE = 5
    BATCH_SIZE = 64
    ENTROPY_COEF = 0.08  
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.01  
    
    # Training parameters
    NUM_EPISODES = 200
    MAX_STEPS_PER_EPISODE = 1000  
    UPDATE_FREQUENCY = 4  
    
    # Paths 
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Logging 
    SAVE_FREQUENCY = 100

    # Screen Recording 
    RECORD_SCREEN = False
    SCREEN_MONITOR = 1  
    SCREEN_FPS = 40
    SCREEN_OUTPUT = os.path.join(RESULTS_DIR, "unity_screen.mp4")
    LOG_FREQUENCY = 10
    VERBOSE = False