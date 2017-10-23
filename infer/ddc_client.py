import xmlrpclib
import sys

s = xmlrpclib.ServerProxy('http://localhost:1337')

artist, title, audio_fp = sys.argv[1:]
diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']

print s.create_chart(artist, title, audio_fp, diffs)
