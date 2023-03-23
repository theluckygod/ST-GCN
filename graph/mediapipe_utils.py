FACE_OFFSET = 0
LEFT_HAND_OFFSET = FACE_OFFSET + 468
POSE_OFFSET = LEFT_HAND_OFFSET + 21
RIGHT_HAND_OFFSET = POSE_OFFSET + 33

lip_landmarks = sorted([61, 185, 40, 39, 37,  0, 267, 269, 270, 409,
                 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])
left_hand_landmarks = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + 21))
right_hand_landmarks = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + 21))
pose_landmarks = list(range(POSE_OFFSET, POSE_OFFSET + 33))


# https://github.com/google/mediapipe/mediapipe/python/solutions/pose_connections.py
POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])


# https://github.com/google/mediapipe/mediapipe/python/solutions/hands_connections.py
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])


# https://github.com/google/mediapipe/mediapipe/python/solutions/face_mesh_connections.py
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)])

FACEMESH_CONTOURS = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL
])

def add_landmarks_to_dict(map_point_dict, landmarks):
    cnt = len(map_point_dict)
    for idx, point in enumerate(landmarks):
        map_point_dict[point] = cnt + idx 
    return map_point_dict

def get_skeleton_edges():
    map_point_dict = dict()
    add_landmarks_to_dict(map_point_dict, lip_landmarks)
    add_landmarks_to_dict(map_point_dict, left_hand_landmarks)
    add_landmarks_to_dict(map_point_dict, pose_landmarks)
    add_landmarks_to_dict(map_point_dict, right_hand_landmarks)

    get_truth_point = lambda p: map_point_dict[p]
    get_truth_connection = lambda connections, offset: [(get_truth_point(u + offset), get_truth_point(v + offset)) for u, v in connections]
    
    return get_truth_connection(FACEMESH_LIPS, FACE_OFFSET) + get_truth_connection(HAND_CONNECTIONS, LEFT_HAND_OFFSET) \
        + get_truth_connection(POSE_CONNECTIONS, POSE_OFFSET) + get_truth_connection(HAND_CONNECTIONS, RIGHT_HAND_OFFSET)



SKELETON_EDGES = get_skeleton_edges()
LANDMARKS = lip_landmarks + left_hand_landmarks + pose_landmarks + right_hand_landmarks

if __name__ == "__main__":
    print("ALL SKELETON_EDGES\n", SKELETON_EDGES)
    print("len:", len(SKELETON_EDGES))