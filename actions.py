from AppOpener import open
from AppOpener import give_appnames
import pyautogui

def handle_action(action_code):
    parts = action_code.split()
    if parts[0] == "OPEN":
        open_App(parts[1].lower())
    elif parts[0] == "CLOSE":
        close_window()
    else:
        print("Action not recognized!")

def open_App(app_name):
    apps = give_appnames()

    found_apps = [app for app in apps if app_name in app.lower()]

    if found_apps:
        open(found_apps[0])
    else:
        print(f"{app_name} Not found")

def close_window():
    pyautogui.hotkey('alt', 'f4')
