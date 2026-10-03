@echo off

echo Starting Voice Command System Server...

start "" http://127.0.0.1:5000

:: Start the wake word engine in its own background terminal window
start cmd /k "python voice_pipeline\wake_word_engine.py"

python main_ui.py

pause