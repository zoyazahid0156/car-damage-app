"""
Vehicle and part labels for the car damage dataset.

Add new vehicle photos here when you expand the dataset.

Example:
    1.jpg - 10.jpg -> Honda Civic 2017 1.8 i-VTEC
    11.jpg - 17.jpg -> Honda Civic 2017 1.5 Turbo

The LABELS section will be filled after we run SAM and inspect
the numbered preview images.
"""


# ============================================================
# VEHICLE / VARIANT INFORMATION
# ============================================================

VARIANTS = {

    # --------------------------------------------------------
    # Honda Civic 2017 - 1.8 i-VTEC
    # --------------------------------------------------------

    "1.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "2.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "3.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "4.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "5.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "6.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "7.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "8.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "9.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },

    "10.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.8 i-VTEC"
    },


    # --------------------------------------------------------
    # Honda Civic 2017 - 1.5 Turbo
    # --------------------------------------------------------

    "11.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },

    "12.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },

    "13.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },

    "14.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },

    "15.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },

    "16.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },

    "17.jpg": {
        "make": "Honda",
        "model": "Civic",
        "year": 2017,
        "variant": "1.5 Turbo"
    },
}


# ============================================================
# PART LABELS
# ============================================================
#
# DO NOT FILL THIS PART YET.
#
# After we run:
#
# python label_car_parts.py masks
#
# SAM will create numbered preview images.
#
# We will then put the numbers here.
#
# Example:
#
# "1.jpg": {
# "hood": [43],
# "front_bumper": [7, 12]
# }
#
# The numbers are the SAM mask numbers visible in the preview.
#

LABELS = {
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_vehicle_info(image_name):
    """
    Return vehicle information for an image.
    """
    if image_name not in VARIANTS:
        raise ValueError(
            f"No vehicle information found for {image_name}. "
            f"Add it to VARIANTS in part_labels.py."
        )

    return VARIANTS[image_name]


def get_variant_folder(image_name):
    """
    Create a safe folder name for the vehicle/variant.
    """

    info = get_vehicle_info(image_name)

    return (
        f"{info['make']}_"
        f"{info['model']}_"
        f"{info['year']}_"
        f"{info['variant'].replace(' ', '_')}"
    )