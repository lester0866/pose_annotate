import cv2


class Rect:
    x = None
    y = None
    w = None
    h = None

    def printit(self):
        print(str(self.x) + ',' + str(self.y) + ',' + str(self.w) + ',' + str(self.h))


class Circle:
    x = None
    y = None
    r = 6
    # Drag in progress
    drag = False
    # already present
    active = True
    # Marker flags by positions
    hold = False
    label = ''

    def __init__(self, label):
        self.label = label

    def printit(self):
        print(str(self.x) + ',' + str(self.y) + ',' + str(self.r))


# endclass

class annots:
    # Limits on the canvas
    keepWithin = Rect()

    # To store rectangle anchor point
    # Here the rect class object is used to store
    # the distance in the x and y direction from
    # the anchor point to the top-left and the bottom-right corner
    selectedJoint = None
    # Selection marker size
    sBlk = 2
    # Whether initialized or not
    initialized = False

    joints = {}

    # Image
    image = None

    # Window Name
    wname = ""

    multiframe = 0
    # Return flag
    returnflag = False
    frame_n = 0
    joints_df = []
    colorDict = {}

    def __init__(self, label):
        self.label = label
        # To store circle
        self.joints[label] = Circle(label)


# endclass


def init(annot_obj, joints, joint_radius, annots_df, windowName, windowWidth, windowHeight, colormap, multiframe):
    # Image
    # annot_obj.image = Img

    # Window name
    annot_obj.wname = windowName
    annot_obj.joints_df = annots_df
    # Limit the selection box to the canvas
    annot_obj.keepWithin.x = 0
    annot_obj.keepWithin.y = 0
    annot_obj.keepWithin.w = windowWidth
    annot_obj.keepWithin.h = windowHeight

    annot_obj.colorDict = colormap
    annot_obj.multiframe = multiframe
    frame_n = 0
    for jt in joints:
        annot_obj(jt)
        # Set rect to zero width and height
        annot_obj.joints[jt].x = 0
        annot_obj.joints[jt].y = 0
        annot_obj.joints[jt].r = joint_radius
        annot_obj.active = True


# enddef

def dragcircle(event, x, y, flags, dragObj):
    # if x < dragObj.keepWithin.x:
    #     x = dragObj.keepWithin.x
    # # endif
    # if y < dragObj.keepWithin.y:
    #     y = dragObj.keepWithin.y
    # # endif
    # if x > (dragObj.keepWithin.x + dragObj.keepWithin.w - 1):
    #     x = dragObj.keepWithin.x + dragObj.keepWithin.w - 1
    # # endif
    # if y > (dragObj.keepWithin.y + dragObj.keepWithin.h - 1):
    #     y = dragObj.keepWithin.y + dragObj.keepWithin.h - 1
    # endif

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseDown(x, y, dragObj)
    # endif
    if event == cv2.EVENT_LBUTTONUP:
        mouseUp(x, y, dragObj)
    # endif
    if event == cv2.EVENT_MOUSEMOVE:
        mouseMove(x, y, dragObj)
    # endif
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseDoubleClick(x, y, dragObj)
    # endif


# enddef

# def pointInRect(pX, pY, rX, rY, rW, rH):
def pointInCircle(pX, pY, rX, rY, rR):
    if ((pX - rX) ** 2) + ((pY - rY) ** 2) < rR ** 2:
        # if rX <= pX <= (rX + rW) and rY <= pY <= (rY + rH):
        return True
    else:
        return False
    # endelseif


# enddef

def updateAnnots(annots_obj, frame_n, im):
    joints = annots_obj.joints.keys()
    annot_df = annots_obj.joints_df[annots_obj.joints_df.frame_n == frame_n][joints]
    if annot_df.empty:
        return
    # This has to be below all of the other conditions
    # if pointInCircle(eX, eY, dragObj.outCircle.x, dragObj.outCircle.y, dragObj.outCircle.r):
    annots_obj.image = im
    annots_obj.frame_n = frame_n
    for joint in annot_df:
        annots_obj.joints[joint].x, annots_obj.joints[joint].y, _score = annot_df[joint].values[0].split('-')

    clearCanvasNDraw(annots_obj)
    return


def mouseDoubleClick(eX, eY, dragObj):
    # if pointInCircle(eX, eY, dragObj.outCircle.x, dragObj.outCircle.y, dragObj.outCircle.r):
    dragObj.returnflag = True
    cv2.destroyWindow(dragObj.wname)
    # endif


# enddef

def mouseDown(x, y, dragObj):
    if dragObj.selectedJoint:

        return

    else:
        for joint_name in dragObj.joints:
            joint = dragObj.joints[joint_name]

            if joint.x == 0:
                continue

            if pointInCircle(x, y, int(joint.x), int(joint.y), joint.r):
                dragObj.selectedJoint = dragObj.joints[joint_name]
                dragObj.selectedJoint.x = x
                dragObj.selectedJoint.y = y
                dragObj.selectedJoint.drag = True
                dragObj.selectedJoint.active = True
                dragObj.selectedJoint.hold = True


# enddef

def mouseMove(eX, eY, dragObj):
    if dragObj.selectedJoint:

        jt = dragObj.selectedJoint
        # jt.x = eX - dragObj.anchor[jt.label].x
        # jt.y = eY - dragObj.anchor[jt.label].y
        jt.x = eX
        jt.y = eY

        if jt.x < dragObj.keepWithin.x:
            jt.x = dragObj.keepWithin.x
        # endif
        if jt.y < dragObj.keepWithin.y:
            jt.y = dragObj.keepWithin.y
        # endif
        if (jt.x + jt.r) > (dragObj.keepWithin.x + dragObj.keepWithin.w - 1):
            jt.x = dragObj.keepWithin.x + dragObj.keepWithin.w - 1 - jt.r
        # endif
        if (jt.y + jt.r) > (dragObj.keepWithin.y + dragObj.keepWithin.h - 1):
            jt.y = dragObj.keepWithin.y + dragObj.keepWithin.h - 1 - jt.r
        # endif

        # update the joint with score 10 since this is done by a human annotator
        if dragObj.multiframe:
            dragObj.joints_df.loc[dragObj.joints_df['frame_n'] >= dragObj.frame_n, jt.label] = str(jt.x) + '-' + str(
                jt.y) + '-10'
        else:
            dragObj.joints_df.loc[dragObj.joints_df['frame_n'] == dragObj.frame_n, jt.label] = str(jt.x) + '-' + str(
                jt.y) + '-10'
        clearCanvasNDraw(dragObj)
        return
    # endif


# enddef

def mouseUp(eX, eY, dragObj):
    if dragObj.selectedJoint:
        dragObj.selectedJoint.drag = False
        disableResizeButtons(dragObj)
        dragObj.selectedJoint.hold = False
        dragObj.selectedJoint.active = False
        dragObj.selectedJoint = None

        # endif

        clearCanvasNDraw(dragObj)


# enddef

def disableResizeButtons(dragObj):
    dragObj.hold = False


# enddef

def clearCanvasNDraw(dragObj):
    # Draw
    tmp = dragObj.image.copy()
    tmp1 = dragObj.image.copy()
    for joint_name in dragObj.joints:
        joint = dragObj.joints[joint_name]
        if joint.x == 0:
            return
        x, y, r = int(joint.x), int(joint.y), int(joint.r)
        cv2.circle(tmp, (x, y), r, dragObj.colorDict[joint_name], -1)
    # apply the overlay
    colorList = [[0, 0, 255], [0, 255, 0], [0, 255, 255]]
    qual = dragObj.joints_df['quality'][dragObj.frame_n]
    cv2.circle(tmp, (10, 10), 10, colorList[qual], -1)
    cv2.addWeighted(tmp, 0.5, tmp1, 0.5, 0, tmp1)
    cv2.imshow(dragObj.wname, tmp1)
    # cv2.waitKey()


# enddef

def debug(df):
    debug_list = []
    col_num = df.columns.get_loc('quality')
    fr_num = df.columns.get_loc('frame_n')
    for i, row in enumerate(df.values):
        quality = row[col_num]
        if quality == -1:
            debug_list.append(row[fr_num])
    return debug_list

# enddef

def debug_save(basepath, debug_list, df, images):
    import cv2
    import pandas as pd
    from collections import deque

    queue = deque(debug_list)
    saved_df = pd.DataFrame(columns=df.columns)
    img_path = 'bootstrap/img/pose_' + basepath
    csv_path = 'bootstrap/csv/pose_' + basepath
    i = 0
    # dequeue
    while queue:
        frame_num = queue.popleft()
        saved_df = saved_df.append(df.iloc[frame_num], ignore_index=False)
        cv2.imwrite(img_path + f'_{frame_num}.png', images[i])
        i += 1
    saved_df.to_csv(csv_path + '.csv', index=False)

    return saved_df

# enddef
