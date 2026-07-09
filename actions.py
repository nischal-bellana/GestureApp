from AppOpener import open
from AppOpener import give_appnames
import pyautogui

def open_App(app_name):
    apps = give_appnames()

    found_apps = [app for app in apps if app_name in app.lower()]

    if found_apps:
        open(found_apps[0])
    else:
        print(f"{app_name} Not found")

def close_window():
    pyautogui.hotkey('alt', 'f4')
