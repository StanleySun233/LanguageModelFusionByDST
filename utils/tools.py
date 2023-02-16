import datetime
import hashlib
import logging
import sys
import time

WARN = 0
INFO = 1
MESSAGE = 2
logging.StreamHandler(sys.stdout)
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().setLevel(logging.INFO)


def logFormat(code, info):
    if code == 0:
        level = 'WARN'
    elif code == 1:
        level = 'INFO'
    elif code == 2:
        level = 'MESSAGE'
    else:
        level = 'MESSAGE'

    t = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    val = '[{}] {} {}'.format(level, t, info)
    logging.info(val)


def getTimeStamp() -> int:
    return int(time.time() * 1000)


def string2sha256(s: str):
    ss = s.encode('utf-8')
    return hashlib.sha256(ss).hexdigest()
