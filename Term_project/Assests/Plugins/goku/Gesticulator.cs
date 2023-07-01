using System;
using System.IO;
using System.Linq;
using UnityEngine;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using Python.Runtime;




namespace Gesticulator
{

    public class PyGesticulatorTestor_Avatar
    {
        // static void AddEnvPath(params string[] paths)
        // {      
        //     var envPaths = Environment.GetEnvironmentVariable("PATH").Split(Path.PathSeparator).ToList();
        //     envPaths.InsertRange(0, paths.Where(x => x.Length > 0 && !envPaths.Contains(x)).ToArray());
        //     Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator.ToString(), envPaths), EnvironmentVariableTarget.Process);
        // }

        static PyGesticulatorTestor_Avatar()
        {
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", @"C:\Users\woduc\Desktop\inde_unity\Assets\CS_DLL\python38.dll", EnvironmentVariableTarget.Process);  
            // Runtime.PythonDLL = @"C:\Users\woduc\Desktop\inde_unity\Assets\CS_DLL\python38.dll";
            var PYTHON_HOME = Environment.ExpandEnvironmentVariables(@"C:\Users\woduc\Anaconda3\envs\gestdio");
            // AddEnvPath(PYTHON_HOME, Path.Combine(PYTHON_HOME, @"Library\bin"));

            Debug.Log("PythonHome ="+PYTHON_HOME);
            PythonEngine.PythonHome = PYTHON_HOME;
            Debug.Log("PythonEngine.PythonPath ="+PythonEngine.PythonPath);
            PythonEngine.PythonPath = string.Join
            (
                Path.PathSeparator.ToString(),
                new string[]
                {
            
                        PythonEngine.PythonPath,
                        Path.Combine(PYTHON_HOME, @"Lib\site-packages"),
                        // @"C:\Users\woduc\Anaconda3\envs\gestdio\Lib\site-packages",
                        // @"C:\Users\woduc\Anaconda3\envs\gestdio\Lib\site-packages\urllib3\util",
                        // @"C:\Users\woduc\Anaconda3\envs\gestdio\lib\site-packages",
                        @"C:\Users\woduc\Anaconda3\envs\gestdio\Lib",
                        // @"C:\Users\woduc\Anaconda3\envs\gestdio\lib\site-packages\urllib3\util",
                        @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master",
                        @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\demo",
                        @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\gesticulator",
                        @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\gesticulator\model",
                        @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\gesticulator\visualization",
                        @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\gesticulator\visualization\motion_visualizer"
                }
            );
        
            Debug.Log(PythonEngine.PythonPath);
            PythonEngine.Initialize();
        }
        public static void main()
        {
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString(@"
                                                import sys;
                                                print('Hello world');
                                                print(sys.version);
                                                import base64;
                                                import datetime;
                                                import json;
                                                import os;
                                                import time;
                                                import traceback;
                                                import urlparse;

                                                import botocore.auth;
                                                import botocore.awsrequest;
                                                import botocore.credentials;
                                                import botocore.endpoint;
                                                import botocore.session;
                                                import boto3.dynamodb.types;
                                                
                                                import ssl;
                ");

                dynamic pysys = Py.Import("sys");   // It uses  PythonEngine.PythonPath 
            
                dynamic pySysPath = pysys.path;
                Debug.Log("pySysPath =" + pySysPath);

                string[] sysPathArray = ( string[]) pySysPath;
                Debug.Log("sysPathArray = " + sysPathArray);
                List<string> sysPath = ((string[])pySysPath).ToList<string>();
                Console.WriteLine("\nsys.path:\n");
                Array.ForEach(sysPathArray, element =>  Console.Write("{0}\t", element));
                dynamic os = Py.Import("os");

                dynamic pycwd = os.getcwd();
                string cwd = (string)pycwd;

                Console.WriteLine("\n\n initial os.cwd={0}", cwd);



                // os.chdir(@"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\demo");
                // pycwd = os.getcwd();
                // cwd = (string)pycwd;
                cwd = @"C:\Users\woduc\Desktop\inde_unity\Assets\Plugins\Winterdust\gesticulator-master\demo";

                Console.WriteLine("\n\n new os.cwd={0}", cwd, "\n\n");
                dynamic avatar = Py.Import("demo.avatar");
                dynamic a = avatar.main();
                // dynamic agent = Py.Import("demo.agent");
                


            } //using Py.GIL
            PythonEngine.Shutdown();
            Console.WriteLine("Press any key...");
            Console.ReadKey();
        } //void main


        // void Add_PySysPath(string path)
        // {
        //     dynamic pysys = Py.Import("sys");   // import sys module from  PythonEngine.PythonPath 
        //     string[] sysPathArray = (string[])pysys.path;
        //     string EnvPath = path;
        //     if (sysPathArray.Contains(EnvPath) == false)
        //         pysys.path.append(EnvPath);
        // }

        // dynamic Get_motionPythonArray()
        // {
        //     dynamic os = Py.Import("os");
        //     dynamic pycwd = os.getcwd();
        //     string cwd = (string)pycwd;
        //     Debug.Log($"[before]cwd:{cwd}");
        //     Add_PySysPath(path: @"C:\Users\SOGANG\Desktop\inde_unity\gesticulator-master");
        //     Add_PySysPath(path: @"C:\Users\SOGANG\Desktop\inde_unity\gesticulator-master\gesticulator\visualization");

        //     dynamic avatar = Py.Import("demo.avatar");
        //     dynamic agent = Py.Import("demo.agent");
            
        //     return null;
        // }

    } //class PyGesticulatorTestor
} //class Gesticulator


