def convert_pixel_distance_to_meter(pixel_distance, reference_height_in_meter, reference_height_in_pixel):
    return (pixel_distance * reference_height_in_meter) / reference_height_in_pixel

def convert_meter_to_pixel_distance(meter, reference_height_in_meter, reference_height_in_pixel):
    return (meter * reference_height_in_pixel) / reference_height_in_meter


    