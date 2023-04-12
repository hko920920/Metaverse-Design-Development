using System.Collections;

using UnityEngine;
using UnityEngine.Networking;

using System;
using System.Net;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;

public class SaveAudio : MonoBehaviour
{
    public AudioSource source;

    public void SaveAudioClip(AudioClip _clip)
    {
        if(source != null)
        {
            this.source.clip = _clip;
            this.source.Play();
        }
        // 소리 저장
        SavWav.Save("/Users/kohankyeong/Documents/2023_MDD_MicRec-main/MDD_MicroPhone/Assets/Scripts/rec", _clip);

        // 저장한 소리를 다시 가져오는 구문



        string FilePath = "/Users/kohankyeong/Documents/2023_MDD_MicRec-main/MDD_MicroPhone/Assets/Scripts/rec.wav";
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
        Debug.Log(text);
        /*
        FileStream fs = new FileStream(Application.persistentDataPath + "/rec.wav",
            FileMode.Open, FileAccess.Read);
        byte[] filedata = new byte[fs.Length];
        fs.Read(filedata, 0, filedata.Length);
        fs.Close();
        */
    }
}
