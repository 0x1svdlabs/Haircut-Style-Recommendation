# Hairstyle recommendation mapping based on face shape.
# Supports 6 face shapes: oval, round, square, heart, diamond, oblong.
# Uses 10 hairstyle categories: Crew Cut, Buzz Cut, Fade Cut, Textured Fringe, 
# French Crop, Side Part, Quiff, Pompadour, Undercut, Slick Back.

RECOMMENDATIONS = {
    "oval": [
        "Crew Cut",
        "Fade Cut",
        "Pompadour",
        "Quiff",
        "Side Part"
    ],
    "round": [
        "Textured Fringe",
        "Fade Cut",
        "French Crop",
        "Quiff",
        "Side Part"
    ],
    "square": [
        "Buzz Cut",
        "Crew Cut",
        "Fade Cut",
        "French Crop",
        "Undercut"
    ],
    "heart": [
        "Textured Fringe",
        "Side Part",
        "Quiff",
        "Undercut",
        "Slick Back"
    ],
    "diamond": [
        "Textured Fringe",
        "Fade Cut",
        "Side Part",
        "Quiff",
        "Slick Back"
    ],
    "oblong": [
        "Crew Cut",
        "Buzz Cut",
        "French Crop",
        "Side Part",
        "Textured Fringe"
    ]
}

def recommend(face_shape):
    """
    Recommend hairstyles based on the predicted face shape.
    
    Args:
        face_shape (str): The predicted face shape label (case-insensitive).
        
    Returns:
        list of str: A list of recommended hairstyle names.
                     Returns a default list if face shape is unrecognized.
    """
    if not face_shape:
        return []
        
    shape_key = face_shape.strip().lower()
    return RECOMMENDATIONS.get(shape_key, ["Crew Cut", "Fade Cut", "Side Part"])
