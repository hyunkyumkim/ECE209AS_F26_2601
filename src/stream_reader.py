"""
MIT License

Copyright (c) 2020 Xander Steenbrugge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/aiXander/Realtime_PyAudio_FFT/tree/master
"""
import time
import sys

import numpy as np

import pyaudio
import sounddevice as sd
from collections import deque

from utils import *




class StreamReaderPyAudio:
    """
    The Stream_Reader continuously reads data from a selected sound source using PyAudio

    Arguments:

        device: int or None:    Select which audio stream to read .
        rate: float or None:    Sample rate to use. Defaults to something supported.
        updatesPerSecond: int:  How often to record new data.

    """

    def __init__(self,
        device = None,
        rate = None,
        updates_per_second  = 1000,
        verbose = False):

        self.rate = rate
        self.verbose = verbose
        self.pa = pyaudio.PyAudio()

        #Temporary variables #hacks!
        self.update_window_n_frames = 1024 #Don't remove this, needed for device testing!
        self.data_buffer = None

        self.device = device
        if self.device is None:
            self.device = self.input_device()
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)

        self.update_window_n_frames = round_up_to_even(self.rate / updates_per_second)
        self.updates_per_second = self.rate / self.update_window_n_frames
        self.info = self.pa.get_device_info_by_index(self.device)
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False
        if self.verbose:
            self.data_capture_delays = deque(maxlen=20)
            self.num_data_captures = 0

        self.stream = self.pa.open(
            input_device_index=self.device,
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input=True,
            frames_per_buffer = self.update_window_n_frames,
            stream_callback=self.non_blocking_stream_read)

        print("\n##################################################################################################")
        print("\nDefaulted to using first working mic, Running on:")
        self.print_mic_info(self.device)
        print("\n##################################################################################################")
        print('Recording from %s at %d Hz\nUsing (non-overlapping) data-windows of %d samples (updating at %.2ffps)'
            %(self.info["name"],self.rate, self.update_window_n_frames, self.updates_per_second))

    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        if self.verbose:
            start = time.time()

        if self.data_buffer is not None:
            self.data_buffer.append_data(np.frombuffer(in_data, dtype=np.int16))
            self.new_data = True

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return in_data, pyaudio.paContinue

    def stream_start(self, data_windows_to_buffer = None):
        self.data_windows_to_buffer = data_windows_to_buffer

        if data_windows_to_buffer is None:
            self.data_windows_to_buffer = int(self.updates_per_second / 2) #By default, buffer 0.5 second of audio
        else:
            self.data_windows_to_buffer = int(data_windows_to_buffer)

        self.data_buffer = NPDataBuffer(self.data_windows_to_buffer, self.update_window_n_frames)

        print("\n-- Starting live audio stream...\n")
        self.stream.start_stream()
        self.stream_start_time = time.time()

    def terminate(self):
        print("Sending stream termination command...")
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def valid_low_rate(self, device, test_rates = [44100, 22050]):
        """Set the rate to the lowest supported audio rate."""
        for testrate in test_rates:
            if self.test_device(device, rate=testrate):
                return testrate

        #If none of the test_rates worked, try the default rate:
        self.info = self.pa.get_device_info_by_index(device)
        default_rate = int(self.info["defaultSampleRate"])

        if self.test_device(device, rate=default_rate):
            return default_rate

        print("SOMETHING'S WRONG! I can't figure out a good sample-rate for DEVICE =>", device)
        return default_rate

    def test_device(self, device, rate=None):
        """given a device ID and a rate, return True/False if it's valid."""
        try:
            self.info = self.pa.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False

            if rate is None:
                rate = int(self.info["defaultSampleRate"])

            stream = self.pa.open(
                format = pyaudio.paInt16,
                channels = 1,
                input_device_index=device,
                frames_per_buffer=self.update_window_n_frames,
                rate = rate,
                input = True)
            stream.close()
            return True
        except Exception as e:
            #print(e)
            return False

    def input_device(self):
        """
        See which devices can be opened for microphone input.
        Return the first valid device
        """
        mics=[]
        for device in range(self.pa.get_device_count()):
            if self.test_device(device):
                mics.append(device)

        if len(mics) == 0:
            print("No working microphone devices found!")
            sys.exit()

        print("Found %d working microphone device(s): " % len(mics))
        for mic in mics:
            self.print_mic_info(mic)

        return mics[0]

    def print_mic_info(self, mic):
        mic_info = self.pa.get_device_info_by_index(mic)
        print('\nMIC %s:' %(str(mic)))
        for k, v in sorted(mic_info.items()):
            print("%s: %s" %(k, v))

class StreamReaderSoundDevice:
    """
    The Stream_Reader continuously reads data from a selected sound source using PyAudio

    Arguments:

        device: int or None:    Select which audio stream to read .
        rate: float or None:    Sample rate to use. Defaults to something supported.
        updatesPerSecond: int:  How often to record new data.

    """

    def __init__(self,
        device = None,
        rate = None,
        verbose = False):

        print("Available audio devices:")
        device_dict = sd.query_devices()
        print(device_dict)

        try:
            sd.check_input_settings(device=device, channels=1, dtype=np.float32, extra_settings=None, samplerate=rate)
        except:
            print("Input sound settings for device %s and samplerate %s Hz not supported, using defaults..." %(str(device), str(rate)))
            rate = None
            device = None

        self.rate = rate
        if rate is not None:
            sd.default.samplerate = rate

        self.device = device
        if device is not None:
            sd.default.device = device

        self.verbose = verbose
        self.data_buffer = None

        # This part is a bit hacky, need better solution for this:
        # Determine what the optimal buffer shape is by streaming some test audio
        self.optimal_data_lengths = []
        with sd.InputStream(samplerate=self.rate,
                            blocksize=0,
                            device=self.device,
                            channels=1,
                            dtype=np.float32,
                            latency='low',
                            callback=self.test_stream_read):
            time.sleep(0.2)

        self.update_window_n_frames = max(self.optimal_data_lengths)
        del self.optimal_data_lengths

        #Alternative:
        #self.update_window_n_frames = round_up_to_even(44100 / updates_per_second)

        self.stream = sd.InputStream(
                                    samplerate=self.rate,
                                    blocksize=self.update_window_n_frames,
                                    device=None,
                                    channels=1,
                                    dtype=np.float32,
                                    latency='low',
                                    extra_settings=None,
                                    callback=self.non_blocking_stream_read)

        self.rate = self.stream.samplerate
        self.device = self.stream.device

        self.updates_per_second = self.rate / self.update_window_n_frames
        self.info = ''
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False
        if self.verbose:
            self.data_capture_delays = deque(maxlen=20)
            self.num_data_captures = 0

        self.device_latency = device_dict[self.device]['default_low_input_latency']

        print("\n##################################################################################################")
        print("\nDefaulted to using first working mic, Running on mic %s with properties:" %str(self.device))
        print(device_dict[self.device])
        print('Which has a latency of %.2f ms' %(1000*self.device_latency))
        print("\n##################################################################################################")
        print('Recording audio at %d Hz\nUsing (non-overlapping) data-windows of %d samples (updating at %.2ffps)'
            %(self.rate, self.update_window_n_frames, self.updates_per_second))

    def non_blocking_stream_read(self, indata, frames, time_info, status):
        if self.verbose:
            start = time.time()
            if status:
                print(status)

        if self.data_buffer is not None:
            self.data_buffer.append_data(indata[:,0])
            self.new_data = True

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return

    def test_stream_read(self, indata, frames, time_info, status):
        '''
        Dummy function to determine what blocksize the stream is using
        '''
        self.optimal_data_lengths.append(len(indata[:,0]))
        return

    def stream_start(self, data_windows_to_buffer = None):
        self.data_windows_to_buffer = data_windows_to_buffer

        if data_windows_to_buffer is None:
            self.data_windows_to_buffer = int(self.updates_per_second / 2) #By default, buffer 0.5 second of audio
        else:
            self.data_windows_to_buffer = int(data_windows_to_buffer)

        self.data_buffer = NPDataBuffer(self.data_windows_to_buffer, self.update_window_n_frames)

        print("\n--🎙  -- Starting live audio stream...\n")
        self.stream.start()
        self.stream_start_time = time.time()

    def terminate(self):
        print("👋  Sending stream termination command...")
        self.stream.stop()
