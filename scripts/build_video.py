"""Silent slide-timing rehearsal aid; students must record their own talk."""
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from src.data import ROOT

if __name__=='__main__':
    build=ROOT/'tmp/deck';slides=sorted(build.glob('slide-??.png'))
    if len(slides)!=9:raise ValueError('Nine inspected slide renders are required')
    # This explicit label appears only in the rehearsal artifact, not research slides.
    frames=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    for i,p in enumerate(slides):
        im=Image.open(p).convert('RGB');draw=ImageDraw.Draw(im)
        draw.rectangle((0,im.height-29,im.width,im.height),fill='white')
        draw.text((12,im.height-25),'SILENT REHEARSAL PREVIEW — Record every student speaking for the submitted presentation',font=font,fill='#526579')
        frame=build/f'preview-{i+1:02d}.png';im.save(frame);frames.append(frame)
    durations=[30,25,35,35,40,30,25,25,15]
    listing=ROOT/'tmp/video_frames.txt'
    listing.write_text(''.join(f"file '{p.as_posix()}'\nduration {d}\n" for p,d in zip(frames,durations))+f"file '{frames[-1].as_posix()}'\n")
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();output=ROOT/'submission/preview_video.mp4'
    subprocess.run([ffmpeg,'-y','-f','concat','-safe','0','-i',str(listing),'-vf','scale=1280:720,format=yuv420p',
                    '-r','10','-c:v','libx264','-preset','fast','-t',str(sum(durations)),'-movflags','+faststart',str(output)],check=True,stdout=subprocess.DEVNULL,stderr=open(ROOT/'tmp/video_build.log','w'))
    probe=subprocess.run([ffmpeg,'-i',str(output)],capture_output=True,text=True)
    (ROOT/'results/video_validation.json').write_text(json.dumps({'duration_target_seconds':sum(durations),'silent':True,'role':'rehearsal aid only','probe':probe.stderr},indent=2))
    print(output,sum(durations),'seconds; silent rehearsal aid')
