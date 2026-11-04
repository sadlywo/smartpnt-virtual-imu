Attribute VB_Name = "Import_Xsens_DOT2_Watch"
Option Explicit

Private swApp As SldWorks.SldWorks
Private logPath As String

Sub main()
    Dim macroPath As String
    Dim macroDir As String
    Dim buildRoot As String
    Dim manifestPath As String
    Dim nativeDir As String
    Dim rows As Collection
    Dim row As Variant
    Dim imported As SldWorks.ModelDoc2
    Dim partPaths As New Collection
    Dim assemblyModel As SldWorks.ModelDoc2
    Dim assemblyDoc As SldWorks.AssemblyDoc
    Dim component As SldWorks.Component2
    Dim dummyComponent As SldWorks.Component2
    Dim templatePath As String
    Dim assemblyPath As String
    Dim saveErrors As Long
    Dim saveWarnings As Long
    Dim ok As Boolean
    Dim dummyComponents(0) As Object

    On Error GoTo Failed
    Set swApp = Application.SldWorks
    swApp.Visible = True
    swApp.UserControl = True
    macroPath = swApp.GetCurrentMacroPathName
    macroDir = Left$(macroPath, InStrRev(macroPath, "\") - 1)
    buildRoot = InputBox( _
        "Select the completed FreeCAD build directory", _
        "Xsens DOT 2 Watch", _
        macroDir _
    )
    If Len(buildRoot) = 0 Then Exit Sub
    If Right$(buildRoot, 1) = "\" Then buildRoot = Left$(buildRoot, Len(buildRoot) - 1)
    manifestPath = buildRoot & "\solidworks_import_manifest.csv"
    If Len(Dir$(manifestPath)) = 0 Then
        Err.Raise vbObjectError + 1, "main", "Missing " & manifestPath
    End If
    logPath = buildRoot & "\solidworks_build_log.txt"
    WriteLog "READ " & manifestPath
    Set rows = ReadManifest(manifestPath)
    nativeDir = buildRoot & "\solidworks"
    EnsureFolder nativeDir
    assemblyPath = nativeDir & "\Xsens_DOT2_Watch.SLDASM"
    If Len(Dir$(assemblyPath)) > 0 Then
        WriteLog "SKIP " & assemblyPath & " reason=already exists"
        Err.Raise vbObjectError + 7, "main", _
            "Assembly already exists; choose a new FreeCAD build directory"
    End If

    For Each row In rows
        Dim stepPath As String
        Dim partPath As String
        stepPath = buildRoot & "\" & Replace(CStr(row(1)), "/", "\")
        partPath = nativeDir & "\" & CStr(row(2))
        If Len(Dir$(partPath)) > 0 Then
            WriteLog "SKIP " & partPath & " reason=already exists"
        Else
            Set imported = ImportStepAsPart(stepPath, partPath)
            swApp.CloseDoc imported.GetTitle
        End If
        partPaths.Add Array( _
            CStr(row(0)), _
            partPath, _
            CDbl(row(3)), _
            CDbl(row(4)), _
            CDbl(row(5)), _
            LCase$(CStr(row(6))) = "true", _
            LCase$(CStr(row(7))) = "true" _
        )
    Next row

    templatePath = swApp.GetUserPreferenceStringValue(swDefaultTemplateAssembly)
    If Len(templatePath) = 0 Then
        Err.Raise vbObjectError + 2, "main", "Default assembly template is not configured"
    End If
    Set assemblyModel = swApp.NewDocument(templatePath, swDwgPaperA4size, 0#, 0#)
    If assemblyModel Is Nothing Then
        Err.Raise vbObjectError + 3, "main", "Cannot create assembly document"
    End If
    Set assemblyDoc = assemblyModel

    For Each row In partPaths
        Set component = AddImportedComponent( _
            assemblyModel, _
            assemblyDoc, _
            CStr(row(1)), _
            CDbl(row(2)), _
            CDbl(row(3)), _
            CDbl(row(4)) _
        )
        If CStr(row(0)) = "DOT2_Fit_Dummy" Then Set dummyComponent = component
    Next row

    assemblyModel.AddConfiguration3 "FIT_CHECK", "Sensor fit inspection", "", 0
    assemblyModel.AddConfiguration3 "PRINT", _
        "Printable assembly without sensor dummy", "", 0
    If dummyComponent Is Nothing Then
        Err.Raise vbObjectError + 4, "main", "DOT2_Fit_Dummy was not inserted"
    End If
    Set dummyComponents(0) = dummyComponent
    ok = assemblyDoc.SetComponentState( _
        swComponentSuppressed, _
        dummyComponents, _
        swSpecifyConfiguration, _
        "PRINT", _
        False _
    )
    If Not ok Then
        Err.Raise vbObjectError + 5, "main", "Cannot suppress DOT2_Fit_Dummy in PRINT"
    End If
    assemblyModel.ShowConfiguration2 "FIT_CHECK"
    ok = assemblyModel.Extension.SaveAs( _
        assemblyPath, _
        0, _
        swSaveAsOptions_Silent, _
        Nothing, _
        saveErrors, _
        saveWarnings _
    )
    If Not ok Or saveErrors <> 0 Then
        Err.Raise vbObjectError + 6, "main", "Assembly save failed"
    End If
    WriteLog "WRITE " & assemblyPath
    MsgBox "SolidWorks assembly created:" & vbCrLf & assemblyPath, vbInformation
    Exit Sub

Failed:
    WriteLog "FAIL main error=" & CStr(Err.Number) & " reason=" & Err.Description
    MsgBox Err.Description, vbCritical
End Sub


Private Function ImportStepAsPart( _
    ByVal stepPath As String, _
    ByVal partPath As String _
) As SldWorks.ModelDoc2
    Dim importData As SldWorks.ImportStepData
    Dim imported As SldWorks.ModelDoc2
    Dim partDoc As SldWorks.PartDoc
    Dim errors As Long
    Dim warnings As Long
    Dim ok As Boolean

    WriteLog "READ " & stepPath
    Set importData = swApp.GetImportFileData(stepPath)
    If importData Is Nothing Then
        WriteLog "FAIL import_options path=" & stepPath
        Err.Raise vbObjectError + 99, "ImportStepAsPart", _
            "Cannot obtain STEP import options: " & stepPath
    End If
    importData.MapConfigurationData = True
    Set imported = swApp.LoadFile4(stepPath, "r", importData, errors)
    If imported Is Nothing Or errors <> 0 Then
        WriteLog "FAIL import_step errors=" & CStr(errors) & " path=" & stepPath
        Err.Raise vbObjectError + 100, "ImportStepAsPart", "STEP import failed: " & stepPath
    End If
    Set partDoc = imported
    errors = partDoc.ImportDiagnosis(True, False, True, 0)
    If errors <> 0 Then
        WriteLog "FAIL import_diagnosis errors=" & CStr(errors) & " path=" & stepPath
        Err.Raise vbObjectError + 101, "ImportStepAsPart", _
            "Import diagnosis failed: " & stepPath
    End If
    ok = imported.Extension.SaveAs( _
        partPath, _
        0, _
        swSaveAsOptions_Silent, _
        Nothing, _
        errors, _
        warnings _
    )
    If Not ok Or errors <> 0 Then
        WriteLog "FAIL save_part errors=" & CStr(errors) & _
            " warnings=" & CStr(warnings) & " path=" & partPath
        Err.Raise vbObjectError + 102, "ImportStepAsPart", "Part save failed: " & partPath
    End If
    WriteLog "WRITE " & partPath
    Set ImportStepAsPart = imported
End Function


Private Function ReadManifest(ByVal path As String) As Collection
    Dim result As New Collection
    Dim handle As Integer
    Dim line As String
    Dim fields As Variant

    handle = FreeFile
    Open path For Input As #handle
    Line Input #handle, line
    Do While Not EOF(handle)
        Line Input #handle, line
        If Len(Trim$(line)) > 0 Then
            fields = Split(line, ",")
            If UBound(fields) <> 7 Then
                Close #handle
                Err.Raise vbObjectError + 20, "ReadManifest", "Invalid CSV row: " & line
            End If
            result.Add fields
        End If
    Loop
    Close #handle
    Set ReadManifest = result
End Function


Private Function AddImportedComponent( _
    ByVal assemblyModel As SldWorks.ModelDoc2, _
    ByVal assemblyDoc As SldWorks.AssemblyDoc, _
    ByVal partPath As String, _
    ByVal xMm As Double, _
    ByVal yMm As Double, _
    ByVal zMm As Double _
) As SldWorks.Component2
    Dim errors As Long
    Dim warnings As Long
    Dim partModel As SldWorks.ModelDoc2

    Set partModel = swApp.OpenDoc6( _
        partPath, swDocPART, swOpenDocOptions_Silent, "", errors, warnings _
    )
    If partModel Is Nothing Or errors <> 0 Then
        Err.Raise vbObjectError + 30, "AddImportedComponent", "Cannot open " & partPath
    End If
    swApp.ActivateDoc3 assemblyModel.GetTitle, True, swUserDecision, errors
    Set AddImportedComponent = assemblyDoc.AddComponent5( _
        partPath, _
        swAddComponentConfigOptions_CurrentSelectedConfig, _
        "", _
        False, _
        "", _
        xMm / 1000#, _
        yMm / 1000#, _
        zMm / 1000# _
    )
    If AddImportedComponent Is Nothing Then
        Err.Raise vbObjectError + 31, "AddImportedComponent", "Cannot insert " & partPath
    End If
    WriteLog "READ " & partPath
End Function


Private Sub EnsureFolder(ByVal path As String)
    If Len(Dir$(path, vbDirectory)) = 0 Then
        MkDir path
        WriteLog "WRITE " & path
    Else
        WriteLog "SKIP " & path & " reason=directory exists"
    End If
End Sub


Private Sub WriteLog(ByVal message As String)
    Dim handle As Integer
    If Len(logPath) = 0 Then Exit Sub
    handle = FreeFile
    Open logPath For Append As #handle
    Print #handle, Format$(Now, "yyyy-mm-dd hh:nn:ss") & " " & message
    Close #handle
    Debug.Print message
End Sub
