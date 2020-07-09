import django
django.setup()

from sefaria.model import *
import sefaria.system.database as database
import os
import sys
import csv
import django

sys.path.insert(1, '/persistent/Sefaria-Project/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'sefaria.settings'
django.setup()

from tqdm import tqdm
from sefaria.model import *
from sefaria.system.database import db

