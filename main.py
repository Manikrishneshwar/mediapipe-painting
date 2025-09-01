import cv2
import mediapipe as mp
import math

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

draw_points=[]
mode= "PENCIL"
ring_finger_up_prev=False  #track prev ring finger state for toggle debounce

brush_color=(0, 0, 255)
brush_size=5
eraser_radius= 20

cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _= frame.shape
    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #get landmarks for pertinent fingers
            index_tip=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_pip=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

            ix, iy=int(index_tip.x * w), int(index_tip.y * h)
            mx, my=int(middle_tip.x * w), int(middle_tip.y * h)
            rix, riy=int(ring_tip.x * w), int(ring_tip.y * h)
            rpipx, rpipy=int(ring_pip.x * w), int(ring_pip.y * h)

            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
            cv2.circle(frame, (mx, my), 10, (0, 255, 0), -1)
            cv2.circle(frame, (rix, riy), 10, (255, 0, 0), -1)  #ring finger assigned to blue to ensure it is seen for toggling modes

            #detecting ring finger up gesture (tip above pip joint)
            ring_finger_up = riy < rpipy - 10  #10px buffer to avoid noise

            #toggle mode only on rising edge of ring finger up
            if ring_finger_up and not ring_finger_up_prev:
                mode= "ERASER" if mode=="PENCIL" else "PENCIL"

            ring_finger_up_prev = ring_finger_up

            #distance between index and middle finger tips
            distance= math.sqrt((mx - ix) ** 2 + (my - iy) ** 2)
            threshold=40

            if mode=="PENCIL":
                if distance>threshold:
                    draw_points.append((ix, iy))
                else:
                    draw_points.append(None)
            else:
                draw_points=[
                    pt for pt in draw_points if pt is None or math.hypot(pt[0] - ix, pt[1] - iy) > eraser_radius
                ]
                draw_points.append(None)

    for i in range(1, len(draw_points)):
        if draw_points[i-1] is not None and draw_points[i] is not None:
            cv2.line(frame, draw_points[i-1], draw_points[i], brush_color, brush_size)

    cv2.putText(
        frame,
        f"Mode: {mode}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        f"Mode: {mode}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Drawing Switching Modes by Ring Finger", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
