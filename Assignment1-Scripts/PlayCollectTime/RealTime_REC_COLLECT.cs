using System.Collections;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

using System;
using System.Net;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;


public class RealTime_REC_COLLECT : MonoBehaviour
{ 
    
    private Text _buttonText;
    private AudioClip mic;

    public AudioSource source;
    public float READ_FLUSH_TIME = 0.5f;
    public ToggleGroup _toggleGroup_micList;

    private List<float> readSamples = new List<float>();

    private bool _isRecord;

    private int lastSample = 0;
    private int channels = 0;
    private int readUpdateId = 0;
    private int previousReadUpdateId = -1;

    private float readFlushTimer = 0.0f;
    private float[] samples = null;

    private void Awake()
    {
        this._buttonText = this.gameObject.GetComponentInChildren<Text>();
    }

    void Start()
    {
        this._isRecord = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (this._isRecord) // 녹음 중이 아니면 실행하지 않는다.
        {
            ReadMic();
            PlayMic();
        }
    }

    public void ShowSelectToggleName()
    {
        if (!Microphone.IsRecording(selectToggle.name))
        {
            this._buttonText.text = "녹음 종료";
            this._isRecord = true;
            DeviceManager.instance.SetAllToggleInteracable(false); // 녹음을 시작하면 Toggle을 조작할 수 없도록 모든 토글을 비활성화하는 함수를 호출
            this.mic = Microphone.Start(selectToggle.name, true, 100, 44100); //You can just give null for the mic name, it's gonna automatically detect the default mic of your system (k)
            this.channels = this.mic.channels; //mono or stereo, for me it's 1 (k)
        }
        else
        {
            this._buttonText.text = "녹음 시작";
            this._isRecord = false;
            DeviceManager.instance.SetAllToggleInteracable(true); // 녹음을 정지하면 다시 Toggle을 조작할 수 있도록 모든 토글을 활성화하는 함수를 호출
            Microphone.End(selectToggle.name);
            this.lastSample = 0; // 샘플 값 초기화
            this.readUpdateId = 0; // 업데이트 값 초기화
            this.previousReadUpdateId = -1; // 이전 업데이트 값 초기화
        }

    }

    private Toggle selectToggle
    {
        get { return this._toggleGroup_micList.ActiveToggles().FirstOrDefault(); }
        // C#의 구문(LINQ)
        // 활성화된 토글을 검색하면서, 가장 처음으로 활성화된 토글을 반환한다.
        // First와 FirstOrDefault의 차이는, 반환 객체의 존재 유무이다. 
        // First는 반환객체가 없으면 오류가 발생하고, FirstOrDefault는 반환이 없어도 됨.
    }
    private void ReadMic()
    {
        int t_pos = Microphone.GetPosition(selectToggle.name);
        int t_diff = t_pos - lastSample;

        if (t_diff > 0)
        {
            this.samples = new float[t_diff * this.channels];
            this.mic.GetData(this.samples, this.lastSample);

            this.readSamples.AddRange(this.samples);//readSamples gonna be converted to an audio clip and be played (k)
        }
        this.lastSample = t_pos;
    }

    private void PlayMic()
    {
        this.readFlushTimer += Time.deltaTime;

        if (this.readFlushTimer > READ_FLUSH_TIME) //0.5f (k)
        {
            if (this.readUpdateId != this.previousReadUpdateId && this.readSamples != null && this.readSamples.Count > 0)
            {
                //Debug.Log("Read happened");
                this.previousReadUpdateId = readUpdateId;

                this.source.clip = AudioClip.Create("Real_time", this.readSamples.Count, this.channels, 44100, false);
                this.source.spatialBlend = 0;//2D sound

                this.source.clip.SetData(this.readSamples.ToArray(), 0); // 읽은 sampledata를 하나의 배열로 만들어 준 뒤 플레이
                
                string FilePath = "YOUR_FILE_NAME"; //?
                FileStream fs = new FileStream(FilePath, FileMode.Open, FileAccess.Read);
                byte[] fileData = new byte[fs.Length];
                fs.Read(fileData, 0, fileData.Length);
                fs.Close();

                string lang = "Kor";    // 언어 코드 ( Kor, Jpn, Eng, Chn )
                string url = $"https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang={lang}";
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
                request.Headers.Add("X-NCP-APIGW-API-KEY-ID", "7ucxs09i5t");
                request.Headers.Add("X-NCP-APIGW-API-KEY", "zcFPMX2LUhHj6fKVDJs2dRpscsXfHMegH44IFIbJ");
                request.Method = "POST";
                request.ContentType = "application/octet-stream";
                request.ContentLength = fileData.Length;
                using (Stream requestStream = request.GetRequestStream())
                {
                    requestStream.Write(fileData, 0, fileData.Length);
                    requestStream.Close();
                }
                HttpWebResponse response = (HttpWebResponse)request.GetResponse();
                Stream stream = response.GetResponseStream();
                StreamReader reader = new StreamReader(stream, Encoding.UTF8);
                string text = reader.ReadToEnd();
                stream.Close();
                response.Close();
                reader.Close();
                Console.WriteLine(text);
                
                if (!this.source.isPlaying)
                {
                    //Debug.Log("Play!");
                    this.source.Play();
                }

                this.readSamples.Clear(); // 소리 출력을 완료했으니 지금까지 읽은 배열을 삭제
                this.readUpdateId++;
            }

            this.readFlushTimer = 0.0f; // 플러시 타이머 초기화
        }
    }
}