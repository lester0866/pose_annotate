#!/usr/bin/env python

import cv2
import numpy as np

from time import sleep

import annotate_tools
import pandas as pd

from tkinter import *
from tkinter import ttk
import os
import glob
from configparser import ConfigParser


def get_filenames():
    path = root_path + "*.avi"
    return [os.path.basename(x) for x in sorted(glob.glob(path))]


def onselect(evt):
    # Note here that Tkinter passes an event object to onselect()
    w = evt.widget
    index = int(w.curselection()[0])
    value = w.get(index)
    print('You selected item %d: "%s"' % (index, value))
    show_video(root_path + value)


def show_video(v_path):
    basepath = os.path.split(v_path)
    player_wname = basepath[1][:-4]
    cv2.destroyAllWindows()
    cv2.namedWindow(player_wname, cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(player_wname, 400, 335)
    cv2.namedWindow(control_wname)
    cv2.moveWindow(control_wname, 400, 50)
    cv2.namedWindow(color_wname)
    cv2.moveWindow(color_wname, 400, 190)
    cap = cv2.VideoCapture(v_path)
    playerwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    playerheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    annot_file = basepath[0] + '/pose_' + basepath[1][:-4] + '.csv'
    annots = pd.read_csv(annot_file)
    annotate_tools.init(annotate_tools.annots, joints, joint_radius, annots, player_wname, playerwidth, playerheight,
                        colorDict, multiframe)
    cv2.setMouseCallback(player_wname, annotate_tools.dragcircle, annotate_tools.annots)
    controls = np.zeros((60, int(playerwidth * 2)), np.uint8)
    text = "=: good, -: bad, 0: no annot, A: prev D: next W: play, S: stop, Q: copy, O: occluded\nZ: save, Esc: quit WITH saving, c: quit WITHOUT saving, X: show # incorrect."
    y0, dy = 20, 25
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(controls,
                    line,
                    (0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    i, x0, y = 0, 0, 20
    x = [0, 85, 270, 400, 510, 680, 800]
    color_map = np.zeros((40, int(playerwidth * 2), 3), np.uint8)
    color_map[:, :] = 255
    for this_joint in joints:
        this_color = colorDict[this_joint]
        this_color = tuple(this_color)
        cv2.putText(color_map, this_joint, (x[i], y), cv2.FONT_HERSHEY_SIMPLEX, 1, this_color, 2)
        i += 1

    tots = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    cv2.createTrackbar('S', player_wname, 0, int(tots) - 1, flick)
    cv2.setTrackbarPos('S', player_wname, 0)
    cv2.createTrackbar('F', player_wname, 1, 100, flick)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    if frame_rate is None:
        frame_rate = 30
    cv2.setTrackbarPos('F', player_wname, frame_rate)
    status = 'stay'

    while True:
        cv2.imshow(control_wname, controls)
        cv2.imshow(color_wname, color_map)
        try:

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = cap.read()
            if im is None:
                break
            r = playerwidth / im.shape[1]
            dim = (int(playerwidth), int(im.shape[0] * r))
            im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow(player_wname, im)
            annotate_tools.updateAnnots(annotate_tools.annots, i, im)

            key = cv2.waitKey(10)
            status = {ord('s'): 'stay', ord('S'): 'stay',
                      ord('w'): 'play', ord('W'): 'play',
                      ord('a'): 'prev_frame', ord('A'): 'prev_frame',
                      ord('d'): 'next_frame', ord('D'): 'next_frame',
                      ord('q'): 'copy', ord('Q'): 'copy',
                      ord('o'): 'occluded', ord('O'): 'occluded',
                      ord('z'): 'save', ord('Z'): 'save',
                      ord('c'): 'quit',
                      ord('0'): 'no_annot',
                      ord('x'): 'incorrect_num',
                      ord('='): 'good',
                      ord('-'): 'bad',
                      255: status,
                      -1: status,
                      27: 'exit'}[key]

            if status == 'play':
                frame_rate = cv2.getTrackbarPos('F', player_wname)
                sleep((0.1 - frame_rate / 1000.0) ** 21021)
                i += 1

                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                continue
            if status == 'stay':
                i = cv2.getTrackbarPos('S', player_wname)
            if status == 'save':
                annots.to_csv(annot_file, index=False)
                print('saved!')
                status = 'stay'
            if status == 'quit':
                print('Quited! Progress NOT saved!')
                break
            if status == 'exit':
                annots.to_csv(annot_file, index=False)
                print('Quited and Saved!')
                break
            if status == 'prev_frame':
                i -= 1
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'occluded':
                joint = annotate_tools.occluded(annotate_tools.annots)
                if joint:
                    print(annots.loc[annots['frame_n'] == i, joint])
                status = 'stay'
            if status == 'next_frame':
                i += 1
                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'copy':
                if i != 0:
                    annots.iloc[i, 4: -1] = annots.iloc[i - 1, 4: -1]
                status = 'stay'
            if status == 'slow':
                frame_rate = max(frame_rate - 5, 0)
                cv2.setTrackbarPos('F', player_wname, frame_rate)
                status = 'play'
            if status == 'fast':
                frame_rate = min(100, frame_rate + 5)
                cv2.setTrackbarPos('F', player_wname, frame_rate)
                status = 'play'
            if status == 'snap':
                cv2.imwrite("./" + "Snap_" + str(i) + ".jpg", im)
                print("Snap of Frame", i, "Taken!")
                status = 'stay'
            if status == 'good':
                annots.loc[annots['frame_n'] == i, 'quality'] = 1
                i += 1
                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'bad':
                annots.loc[annots['frame_n'] == i, 'quality'] = -1
                i += 1
                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'no_annot':
                annots.loc[annots['frame_n'] == i, 'quality'] = 0
                i += 1
                if i == tots:
                    i = 0
                cv2.setTrackbarPos('S', player_wname, i)
                status = 'stay'
            if status == 'incorrect_num':
                debug_list = annotate_tools.debug(annots)
                print('num_incorrect:' + str(len(debug_list)))
                status = 'stay'
        except KeyError:
            print("Invalid Key was pressed")
        except ValueError:
            print("Don't try going out of the box!")
            break

    cv2.destroyWindow(player_wname)
    cv2.destroyWindow(control_wname)
    cv2.destroyWindow(color_wname)


# Define the drag object

config = ConfigParser()
config.read('config.ini')

cfg = config.get('configsection', 'config')
root_path = config.get(cfg, 'dataPath')
bt_path = config.get(cfg, 'btPath')
joints = config.get(cfg, 'joints').split(', ')
joint_radius = int(config.get(cfg, 'joint_radius'))
multiframe = int(config.get(cfg, 'multiframe'))


def flick(x):
    pass


def extract_frames():
    from collections import deque
    import imageio
    # handling image files
    csv_list = [os.path.basename(x) for x in sorted(glob.glob(root_path + '*.csv'))]
    img_list = []

    print('Merging Files...')
    final_df = None
    for csv in csv_list:
        df = pd.read_csv(root_path + csv)
        debug_list = annotate_tools.debug(df)
        queue = deque(debug_list)
        this_df = pd.DataFrame(columns=df.columns)
        # dequeue

        v_path = root_path + csv[5:-4] + '.avi'
        cap = cv2.VideoCapture(v_path)
        while queue:
            frame_num = queue.popleft()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, im = cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            this_df = this_df.append(df.loc[df['frame_n'] == frame_num], ignore_index=False)
            img_list.append(im)
        if final_df is None:
            final_df = this_df
        else:
            final_df = final_df.append(this_df, ignore_index=True)

    print('Finished Merging')
    print('Spliting Files...')
    images = np.stack(img_list, axis=0).astype(np.uint8)
    video_list = np.array_split(images, 5)
    counter = 1
    folder_name = root_path.split('/')[-2]

    for video in video_list:
        export_path = bt_path + f'{counter}/{folder_name}'
        imageio.mimwrite(export_path + '.avi', video, fps=30, macro_block_size=1)
        counter += 1

    df_list = []
    start = 0
    for i in range(0, len(video_list)):
        end = video_list[i].shape[0] + start
        df_list.append(final_df.loc[start:end - 1, :])
        start = end

    counter = 1
    for i in range(len(df_list)):
        index = 0
        for j, row in df_list[i].iterrows():
            df_list[i].at[j, 'frame_n'] = index
            index += 1
        export_path = bt_path + f'{counter}/pose_{folder_name}'
        df_list[i].to_csv(export_path + '.csv', index=False)
        counter += 1
    # end csv handling
    print('Finished Splitting')
    print('Loading Files...')
    # load()
    print('Loading Successful! Done with this patient!')


def load():
    folder_name = root_path.split('/')[-2]
    for counter in range(1, 6):
        global l
        export_path = bt_path + f'{counter}/{folder_name}'
        exists = os.path.isfile(export_path + '.avi')
        if exists:
            l.insert(END, export_path + '.avi')
        else:
            raise Exception("CANNOT LOAD - FILE DOES NOT EXIST!")


player_wname = 'video'
control_wname = 'controls'
color_wname = 'color list'
NUM_COLORS = len(joints)
colorList = [[0, 0, 255], [0, 255, 170], [0, 170, 255], [0, 255, 0], [255, 0, 170], [255, 255, 0], [255, 0, 0]]
colorDict = dict(zip(joints, colorList))
root = Tk()
l = Listbox(root, selectmode=SINGLE, height=30, width=60)
l.grid(column=0, row=0, sticky=(N, W, E, S))
s = ttk.Scrollbar(root, orient=VERTICAL, command=l.yview)
s.grid(column=1, row=0, sticky=(N, S))
l['yscrollcommand'] = s.set
ttk.Sizegrip().grid(column=1, row=1, sticky=(S, E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.geometry('350x500+50+50')
root.title('Select Video')
button = Button(text="Split + Reload", command=extract_frames)
button.grid(column=0, row=1, sticky=(W + S + N + E))
for filename in get_filenames():
    l.insert(END, filename)

l.bind('<<ListboxSelect>>', onselect)

root.mainloop()
