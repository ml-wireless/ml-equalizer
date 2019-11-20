import equalizer.util.offline as offline

# params for channel simulator
pream_size = 50
model_tap_size = 2
data_size = 50000
train_snr = 20

"""
Compute nth-step lms output

@param symbol complex sample of received signal
@param weights current lms FIR weights
@return nth-step lms output
"""
#def lms(symbol, weights):
#    return 

if __name__ == "__main__":
    # gen received symbols using channel simulator
    pream, tap, recv = offline.gen_ktap(data_size,
            pream_size, model_tap_size, train_snr)
    print("pream:",pream)
    print("tap:",tap)
    print("recv:",recv)
