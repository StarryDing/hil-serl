def get_snr(sigmas):
    return sigmas**-2

def get_ve_weightings(weight_schedule, snrs, sigma_data):
    """
    Get the weightings for the loss function.
    在VE风格的加噪中, 不同噪声水平下计算的 mse loss 也应该有不同的权重
    Args:
        weight_schedule: The weight schedule.
        snrs: The SNRs.
        sigma_data: The sigma data.
    Returns:
        The weightings.
    """
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    else:
        raise NotImplementedError()
    return weightings