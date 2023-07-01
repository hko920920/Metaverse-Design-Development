﻿using UnityEngine;
using AgentGesticulator;
using System.Collections.Generic; // So we can use List<>
using System.Collections;
using FrostweepGames.Plugins.Native;
using UnityEngine.Audio;
using System;
using System.IO;

// [RequireComponent(typeof(AudioSource))]

public class Agentaudio : MonoBehaviour 
{
	
	// public float minThreshold = 0;
	// public float frequency = 0.0f;
	// public int audioSampleRate = 44100;
	// public string microphone;
	// public FFTWindow fftWindow;
	// public Dropdown micDropdown;
	// public Slider thresholdSlider;

	// private List<string> options = new List<string>();
	// private int samples = 8192; 
	public AudioSource audioSource;
    public AudioClip audioClip;
    // public AudioMixer masterMixer;

    // public Button startRecordButton, stopRecordButton, playRecordedAudioButton, exportAudioButton;


    public void Start() {
        // RequestPermission();

        // startRecordButton.onClick.AddListener(Record);
        // stopRecordButton.onClick.AddListener(Stop);
        // playRecordedAudioButton.onClick.AddListener(PlayRecordedAudio);

        //get components you'll need
        audioSource = GetComponent<AudioSource>();
		PyGesticulatorTestor_Agent.main();
		Invoke("PlayRecordedAudio", 0.5f);

		// get all available microphones
		// foreach (string device in Microphone.devices) {
		// 	if (microphone == null) {
		// 		//set default mic to first mic found.
		// 		microphone = device;
		// 	}
		// 	options.Add(device);
		// }
		// microphone = options[PlayerPrefsManager.GetMicrophone ()];
		// minThreshold = PlayerPrefsManager.GetThreshold ();

		//add mics to dropdown
		// micDropdown.AddOptions(options);
		// micDropdown.onValueChanged.AddListener(delegate {
		// 	micDropdownValueChangedHandler(micDropdown);
		// });

		// thresholdSlider.onValueChanged.AddListener(delegate {
		// 	thresholdValueChangedHandler(thresholdSlider);
		// });
		//initialize input with default mic
		// UpdateMicrophone ();
	}

	public void Update()
	{
		// if (Input.GetKeyDown(KeyCode.Space))
		// {
		// 	microphone = options[PlayerPrefsManager.GetMicrophone ()];
		// 	Record();
		// 	Debug.Log("key down");
			
		// }

		// if(Input.GetKeyUp(KeyCode.Space))
		// {
		// 	Stop();
		// 	ExportAudio();
		// 	Debug.Log("key up");

		// }
	} 
	
    public void PlayRecordedAudio()
    {
        if (audioClip == null)
            return;

        audioSource.clip = audioClip;
        audioSource.Play();
    }

    // public void Stop()
    // {
    //    // audioSource.Stop();

    //     if (!HasConnectedMicDevices())
    //         return;
    //     if (!IsRecordingNow(microphone))
    //         return;

    //     //audioSource.Stop();
    //     Microphone.End(microphone);
    // }

    // public void Record()
    // {
    //     if (!HasConnectedMicDevices())
	// 	{
	// 		Debug.Log("error");
    //         return;
	// 	}
    //     UpdateMicrophone();
    //     Debug.Log("recording started with " + microphone);
    //     audioSource.clip = audioClip;
    //     // audioSource.Play();
    //     //audioSource.loop = true;
    // }

    // public void OnApplicationQuit()
    // {
    //     Application.Quit();
    // }

    // public void MonitorAudioOn()
    // {
    //     masterMixer.SetFloat("MonitorInstr", 0.10f);
    //     masterMixer.SetFloat("MonitorMic", 0.10f);
    // }

    // public void MonitorAudioOff()
    // {
    //     masterMixer.SetFloat("MonitorInstr", 0);
    //     masterMixer.SetFloat("MonitorMic", 0);
    // }

    // public void ExportAudio()
    // {
    //     SavWav.Save("UnityTest", audioClip);
		
    // }

    // public void RequestPermission()
    // {
    //     CustomMicrophone.RequestMicrophonePermission();
    // }

	void UpdateMicrophone(){
		//audioSource.Stop(); 
		//Start recording to audioclip from the mic
		audioClip = Resources.Load("Agentaudio/audio2.wav") as AudioClip;
        //audioSource.clip = audioClip;
		//audioSource.loop = true; 
		// Mute the sound with an Audio Mixer group becuase we don't want the player to hear it
		Debug.Log("Agent is speaking");

		/*if (Microphone.IsRecording (microphone)) { //check that the mic is recording, otherwise you'll get stuck in an infinite loop waiting for it to start
			while (!(Microphone.GetPosition (microphone) > 0)) {
			} // Wait until the recording has started. 
		
			Debug.Log ("recording started with " + microphone);

			// Start playing the audio source
			audioSource.Play (); 
		} else {
			//microphone doesn't work for some reason

			Debug.Log (microphone + " doesn't work!");
		}*/
	}

    // public static bool HasConnectedMicDevices()
    // {
    //     return Microphone.devices.Length > 0;
    // }

    // public static bool IsRecordingNow(string deviceName)
    // {
    //     return Microphone.IsRecording(deviceName);
    // }



    // public void micDropdownValueChangedHandler(Dropdown mic){
	// 	microphone = options[mic.value];
	// 	UpdateMicrophone ();
	// }

	// public void thresholdValueChangedHandler(Slider thresholdSlider){
	// 	minThreshold = thresholdSlider.value;
	// }
	
	// public float GetAveragedVolume()
	// { 
	// 	float[] data = new float[256];
	// 	float a = 0;
	// 	audioSource.GetOutputData(data,0);
	// 	foreach(float s in data)
	// 	{
	// 		a += Mathf.Abs(s);
	// 	}
	// 	return a/256;
	// }
	
	// public float GetFundamentalFrequency()
	// {
	// 	float fundamentalFrequency = 0.0f;
	// 	float[] data = new float[samples];
	// 	audioSource.GetSpectrumData(data,0,fftWindow);
    //     float s = 0.0f;
	// 	int i = 0;
	// 	for (int j = 1; j < samples; j++)
	// 	{
	// 		if(data[j] > minThreshold) // volumn must meet minimum threshold
	// 		{
	// 			if ( s < data[j] )
	// 			{
	// 				s = data[j];
	// 				i = j;
	// 			}
	// 		}
	// 	}
	// 	fundamentalFrequency = i * audioSampleRate / samples;
	// 	frequency = fundamentalFrequency;
    //     return fundamentalFrequency;
	// }
}