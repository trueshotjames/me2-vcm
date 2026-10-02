# voice_pipeline/intent_parser.py

def parse_text(text):
    # Clean the incoming text
    text = text.replace('[BLANK]', '').replace('[blank]', '').strip().lower()
    
    # Remove common punctuation so variations match perfectly
    text = text.replace('?', '').replace('.', '')

    # Complete Lookup Table (Variations x Values)
    commands = {
        # --- FIXED COMMANDS ---
        
        # PLAY_MUSIC
        "play music": ("music_play", None),
        "start music": ("music_play", None),
        "play some music": ("music_play", None),
        
        # WEATHER
        "weather": ("weather", None),
        "what's the weather": ("weather", None),
        "tell me the weather": ("weather", None),
        
        # TIME
        "time": ("time", None),
        "what time is it": ("time", None),
        "tell me the time": ("time", None),
        
        # LIGHT_ON
        "lights on": ("toggle", None),
        "power on the lights": ("toggle", None),
        "turn on the lights": ("toggle", None),
        
        # LIGHT_OFF
        "lights out": ("toggle", None),
        "kill the lights": ("toggle", None),
        "shut off the lights": ("toggle", None),
        
        # PAUSE
        "pause": ("music_pause", None),
        "pause audio": ("music_pause", None),
        "pause song": ("music_pause", None),
        
        # STOP
        "stop": ("music_stop", None),
        "stop playing": ("music_stop", None),
        "end playback": ("music_stop", None),
        
        # NEXT
        "next song": ("music_skip", None),
        "skip song": ("music_skip", None),
        "play next song": ("music_skip", None),
        
        # VOLUME_UP
        "volume up": ("music_vol_up", None),
        "increase the volume": ("music_vol_up", None),
        "turn the volume up": ("music_vol_up", None),
        
        # VOLUME_DOWN
        "volume down": ("music_vol_down", None),
        "lower the volume": ("music_vol_down", None),
        "turn the volume down": ("music_vol_down", None),
        
        # CALL 
        "call": ("call_mom", None),
        "make a call": ("call_mom", None),
        "make a phone call": ("call_mom", None),
        
        # MESSAGE
        "message": ("message", None),
        "send a message": ("message", None),
        "send my message": ("message", None),
        
        # LIST_REMINDERS
        "reminders": ("get_reminders", None),
        "show my reminders": ("get_reminders", None),
        "list my reminders": ("get_reminders", None),
        

        # --- SLOTTED COMMANDS ---
        
        # TIMER
        "timer ten seconds": ("timer", {"minutes": 0, "seconds": 10}),
        "countdown for ten seconds": ("timer", {"minutes": 0, "seconds": 10}),
        "start a timer for ten seconds": ("timer", {"minutes": 0, "seconds": 10}),
        
        "timer thirty seconds": ("timer", {"minutes": 0, "seconds": 30}),
        "countdown for thirty seconds": ("timer", {"minutes": 0, "seconds": 30}),
        "start a timer for thirty seconds": ("timer", {"minutes": 0, "seconds": 30}),
        
        "timer one minute": ("timer", {"minutes": 1, "seconds": 0}),
        "countdown for one minute": ("timer", {"minutes": 1, "seconds": 0}),
        "start a timer for one minute": ("timer", {"minutes": 1, "seconds": 0}),
        
        # ALARM - 06:00:00
        "alarm six o'clock am": ("alarm", "06:00:00"),
        "wake me up at six o'clock am": ("alarm", "06:00:00"),
        "set an alarm for six o'clock am": ("alarm", "06:00:00"),
        
        "alarm six am": ("alarm", "06:00:00"),
        "wake me up at six am": ("alarm", "06:00:00"),
        "set an alarm for six am": ("alarm", "06:00:00"),

        # ALARM - 08:00:00
        "alarm eight o'clock am": ("alarm", "08:00:00"),
        "wake me up at eight o'clock am": ("alarm", "08:00:00"),
        "set an alarm for eight o'clock am": ("alarm", "08:00:00"),
        
        "alarm eight am": ("alarm", "08:00:00"),
        "wake me up at eight am": ("alarm", "08:00:00"),
        "set an alarm for eight am": ("alarm", "08:00:00"),

        # ALARM - 21:00:00
        "alarm nine o'clock pm": ("alarm", "21:00:00"),
        "wake me up at nine o'clock pm": ("alarm", "21:00:00"),
        "set an alarm for nine o'clock pm": ("alarm", "21:00:00"),
        
        "alarm nine pm": ("alarm", "21:00:00"),
        "wake me up at nine pm": ("alarm", "21:00:00"),
        "set an alarm for nine pm": ("alarm", "21:00:00"),
        
        # TEMPERATURE 
        "temperature eighteen degrees": ("temp", 18),
        "change the temperature to eighteen degrees": ("temp", 18),
        "set the temperature to eighteen degrees": ("temp", 18),
        
        "temperature twenty two degrees": ("temp", 22),
        "change the temperature to twenty two degrees": ("temp", 22),
        "set the temperature to twenty two degrees": ("temp", 22),
        
        "temperature twenty six degrees": ("temp", 26),
        "change the temperature to twenty six degrees": ("temp", 26),
        "set the temperature to twenty six degrees": ("temp", 26),
        
        # BRIGHTNESS 
        "brightness twenty percent": ("dim", 20),
        "adjust brightness to twenty percent": ("dim", 20),
        "brightness level twenty percent": ("dim", 20),
        
        "brightness sixty percent": ("dim", 60),
        "adjust brightness to sixty percent": ("dim", 60),
        "brightness level sixty percent": ("dim", 60),
        
        "brightness one hundred percent": ("dim", 100),
        "adjust brightness to one hundred percent": ("dim", 100),
        "brightness level one hundred percent": ("dim", 100),
        
        # COLOR 
        "change color to red": ("color", "red"),
        "switch color to red": ("color", "red"),
        "set color to red": ("color", "red"),
        
        "change color to blue": ("color", "blue"),
        "switch color to blue": ("color", "blue"),
        "set color to blue": ("color", "blue"),
        
        "change color to green": ("color", "green"),
        "switch color to green": ("color", "green"),
        "set color to green": ("color", "green"),
        
        # CREATE_REMINDER
        "reminder drink water": ("reminder", "drink water"),
        "remind me to drink water": ("reminder", "drink water"),
        "create a reminder to drink water": ("reminder", "drink water"),
        
        "reminder study": ("reminder", "study"),
        "remind me to study": ("reminder", "study"),
        "create a reminder to study": ("reminder", "study"),
        
        "reminder exercise": ("reminder", "exercise"),
        "remind me to exercise": ("reminder", "exercise"),
        "create a reminder to exercise": ("reminder", "exercise")
    }
    
    # Direct Lookup
    if text in commands:
        return commands[text]
        
    return "Unrecognized", None