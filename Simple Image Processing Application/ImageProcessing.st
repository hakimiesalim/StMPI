<?xml version="1.0"?>

<st-source>
<time-stamp>From VisualWorks® Personal Use Edition, 8.3 of July 28, 2017 on February 22, 2020 at 10:42:48 AM</time-stamp>


<class>
<name>ImageProcessing</name>
<environment>Smalltalk</environment>
<super>UI.ApplicationModel</super>
<private>false</private>
<indexed-type>none</indexed-type>
<inst-vars>numproc ProcessingTime </inst-vars>
<class-inst-vars></class-inst-vars>
<imports></imports>
<category></category>
<attributes>
<package>ImageProcessingApp</package>
</attributes>
</class>

<shared-variable>
<name>PT</name>
<environment>ImageProcessing</environment>
<private>false</private>
<constant>false</constant>
<category>PT</category>
<attributes>
<package>ImageProcessingApp</package>
</attributes>
</shared-variable>

<shared-variable>
<name>intercommSobelPtr</name>
<environment>ImageProcessing</environment>
<private>false</private>
<constant>false</constant>
<category>intercommSobelPtr</category>
<attributes>
<package>ImageProcessingApp</package>
</attributes>
</shared-variable>

<!-- -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -->


<methods>
<class-id>ImageProcessing class</class-id> <category>interface specs</category>

<body package="ImageProcessingApp" selector="windowSpec">windowSpec
</methods>

<!-- -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -->


<methods>
<class-id>ImageProcessing</class-id> <category>actions</category>

<body package="ImageProcessingApp" selector="ApplySobel">ApplySobel

<body package="ImageProcessingApp" selector="ChangeParallelTechnique">ChangeParallelTechnique

<body package="ImageProcessingApp" selector="CloseImage">CloseImage

<body package="ImageProcessingApp" selector="DecreaseBrightness">DecreaseBrightness

<body package="ImageProcessingApp" selector="IncreaseBrightness">IncreaseBrightness

<body package="ImageProcessingApp" selector="LoadImage">LoadImage

<body package="ImageProcessingApp" selector="SaveImage">SaveImage

<body package="ImageProcessingApp" selector="UndoSobel">UndoSobel

<body package="ImageProcessingApp" selector="UseCudaButton">UseCudaButton

<body package="ImageProcessingApp" selector="UseMPIButton">UseMPIButton
</methods>

<methods>
<class-id>ImageProcessing</class-id> <category>aspects</category>

<body package="ImageProcessingApp" selector="ProcessingTime">ProcessingTime

<body package="ImageProcessingApp" selector="numproc">numproc
</methods>

<methods>
<class-id>ImageProcessing</class-id> <category>initialize-release</category>

<body package="ImageProcessingApp" selector="initialize">initialize
</methods>

<methods>
<class-id>ImageProcessing</class-id> <category>changing</category>

<body package="ImageProcessingApp" selector="changeRequest">changeRequest
</methods>

</st-source>