from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import obspy 
from scipy.signal import detrend
from scipy.signal import spectrogram

def fk_filter(file,data, fs, ch_space, mask=False, max_wavenum=0.1, min_wavenum=0.01, max_freq=100., min_freq=0.01, plot=True):
    """FK filter for a 2D DAS numpy array. Returns a filtered image."""
    
    import numpy as np
    
    # # frequencies
    # nx,nt = np.shape(data)
    # dt    = 1./fs
    # nyq_f = nt//2
    # f = np.fft.fftfreq(nt, d=dt)[slice(0,nyq_f)]
    
    # # wavenumbers
    # dx = ch_space
    # nyq_k = nx//2
    # k = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    
    
    # # frequency-wavenumber power spectral density
    # B = np.fft.fftshift(np.fft.fft2(data/1e9),axes=0)
    # A = B[:,slice(0,nyq_f)]
    # sp2 = 2*(np.abs(A)**2) / (nt**2)
    # sp2 = 10*np.log10(sp2)
    # '''
    # if mask:
    #     freqsgrid=np.broadcast_to(f,fftdata.shape)   
    #     wavenumsgrid=np.broadcast_to(k,fftdata.T.shape).T
    
    #     # Define mask and blur the edges 
    #     mask=np.logical_and(np.logical_and(np.logical_and(\
    #         abs(wavenumsgrid)<=max_wavenum,\
    #         abs(wavenumsgrid)>min_wavenum),\
    #         abs(freqsgrid)<max_freq),\
    #         abs(freqsgrid)>min_freq)
    #     x=mask*1.
    #     blurred_mask = ndimage.gaussian_filter(x, sigma=3)
    
    #     # Apply the mask to the data
    #     sp2 = sp2 * blurred_mask
    # '''    
    # ftimagep = np.fft.ifftshift(B, axes=0)
    # # Finally, take the inverse transform
    # imagep = np.fft.ifft2(ftimagep)
    # imagep = imagep.real



    # Read the SEG-Y file
    st = obspy.read(file, format='SEGY')

    # Convert the data to a NumPy array
    trace_array = np.vstack([tr.data for tr in st])

    # Optionally, crop the trace data to focus on a specific portion
    trace_array_cropped = trace_array[:, 0:1000]  # Adjust as needed

    # Perform 2D Fourier Transform (FK Transform)
    fk_domain = fftshift(fft2(trace_array_cropped))

    # Create the frequency and wavenumber axes
    n_traces, n_samples = trace_array_cropped.shape
    dx = ch_space  # Spatial sampling interval (distance between traces)
    dt = st[0].stats.delta  # Time sampling interval
    kx = np.fft.fftfreq(n_traces, d=dx)  # Wavenumber axis
    f = np.fft.fftfreq(n_samples, d=dt)  # Frequency axis

    # Create a meshgrid for plotting
    KX, F = np.meshgrid(kx, f)

    # Apply an FK filter (e.g., bandpass filter in FK domain)
    fk_filter = np.ones_like(fk_domain)
    # Example: Zero out low wavenumbers and frequencies
    low_f_cutoff = 0.1  # Frequency cutoff
    high_f_cutoff = 0.4  # Frequency cutoff
    low_kx_cutoff = 0.1  # Wavenumber cutoff
    high_kx_cutoff = 0.4  # Wavenumber cutoff

    # Apply a simple band-pass filter in the FK domain
    fk_filter[(np.abs(F) < low_f_cutoff) | (np.abs(F) > high_f_cutoff)] = 0
    fk_filter[(np.abs(KX) < low_kx_cutoff) | (np.abs(KX) > high_kx_cutoff)] = 0

    # Apply the FK filter to the FK domain data
    fk_filtered = fk_domain * fk_filter

    # Inverse 2D Fourier Transform to get back to the time-space domain
    filtered_trace_array = np.real(ifft2(fftshift(fk_filtered)))
    
    if plot==True:
        vMine = np.percentile(sp2,.10)
        vMaxe = np.percentile(sp2,99.90)    
        clim=[vMine,vMaxe]
        
        fig,ax = plt.subplots(figsize=(10,6))        
        plt.imshow(sp2.T,extent=[max(k),min(k),min(f),max(f),],
            aspect='auto',cmap='magma',interpolation=None,origin='lower',
            vmin=clim[0],vmax=clim[1])
        h = plt.colorbar()
        h.set_label('Power Spectra [dB] (rel. 1 $(\epsilon/s)^2$)')
        plt.ylabel('frequency [1/s]')
        plt.xlabel('wavenumber [1/m]')
        # Add Velocity lines
        c=-1000; ax.plot(k,k*c,color='w',linestyle='--',label='1km/s');
        c=-2000; ax.plot(k,k*c,color='gray',linestyle='--',label='2km/s');
        c=-4000; ax.plot(k,k*c,color='k',linestyle='--',label='4km/s');
        plt.ylim([min(f),max(f)])
        plt.xlim([min(k),max(k)])
        plt.ylim([1,250])
        plt.xlim([-0.25,0.25])
        
        
        plt.legend()
        plt.tight_layout()
        outfile = 'FK_test.png'
        plt.savefig(outfile, format='png', dpi=300)
        plt.show()    
        
    return f,k[2:],sp2.T, imagep.T

def bp_filter(data, freqmin, freqmax, df, corners=4, zerophase=False):
    from scipy.signal import iirfilter
    
    try:
        from scipy.signal import sosfilt
        from scipy.signal import zpk2sos
    except ImportError:
        from ._sosfilt import _sosfilt as sosfilt
        from ._sosfilt import _zpk2sos as zpk2sos
    #Taper?
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis=1)
        return sosfilt(sos, firstpass[::-1], axis=1)[::-1]
    else:
        return sosfilt(sos, data, axis=1)
    
def medianSubtract(stream):
    import numpy as np
    arr = stream2array(stream)
    med = np.median(arr,axis=0)
    arr = arr-med
    return array2stream(arr,stream.copy())


if __name__ == "__main__":

    file = 'test1'
    # Choose filter fk, bp, or none

    filt            =   'fk'
    # Path to your SEG-Y file
    segy_file = 'C:/Projects/RTE/SEGY/stacked_rigshot.sgy'

    # Read the SEG-Y file
    st = obspy.read(segy_file, format='SEGY',unpack_trace_headers=True)
    # f-k filter
    max_wavenum     =   0.00
    min_wavenum     =   0.0   
    max_freq        =   500
    min_freq        =   20
     
    startCh            = 10
    endCh              = 1900
        
    startt              = 0
    endt                = .4
    
    st_trimmed = st[startCh:endCh]



    if len(st_trimmed) > 0:
        trace_array_cropped = np.vstack([tr.data for tr in st_trimmed])
    else:
        trace_array_cropped = np.empty((0, 0))  # Handle case where no traces are found


    x1 = st[startCh].stats.segy.trace_header['group_coordinate_x']
    x2 = st[endCh].stats.segy.trace_header['group_coordinate_x']
    y1 = st[startCh].stats.segy.trace_header['group_coordinate_y']
    y2 = st[endCh].stats.segy.trace_header['group_coordinate_y']
   

    #channel_spacing     = (np.sqrt(((x2-x1)**2)+((y2-y1)**2))/(endCh-startCh))/10
    channel_spacing     =  1.09
    samplingFrequency   = st[0].stats.sampling_rate
 
            
    tspan               = endt - startt
    nsamps              = int((endt-startt)*samplingFrequency)
    
    startSample         = int(startt*samplingFrequency)
    endSample           = int(endt*samplingFrequency)
    
    # Select portion of data of interest
    acousticData = trace_array_cropped[:,startSample:endSample]
    
    acousticData = acousticData
    print(np.shape(acousticData))
    
    # Median filter to remove noise across all channels
    #med = np.median(acousticData,axis=0)
    #acousticData = acousticData-med
    # Remove linear trend
    acousticData = detrend(acousticData, axis=1, type='linear', bp=0, overwrite_data=True)

    
    if filt=='fk':
        f,k,psd,imagep = fk_filter(file,acousticData, samplingFrequency, channel_spacing, mask=True, max_wavenum=max_wavenum, min_wavenum=min_wavenum, max_freq=max_freq, min_freq=min_freq, plot=True)
        
 

    plt.imshow(acousticData)

    plt.show()
