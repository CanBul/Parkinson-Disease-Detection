import os
from pydub import AudioSegment
from pydub.playback import play
import easygui


wavFiles = [f for f in os.listdir('./') if f.endswith('.wav')]
for each in wavFiles:
    play(AudioSegment.from_wav(each))
    keep = easygui.ynbox('Are we keeping this?',
                         str(each[:-4]), ('Keep', 'Delete'))
    if keep:
        continue
    else:
        os.remove(each)