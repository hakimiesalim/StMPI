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

<body package="ImageProcessingApp" selector="windowSpec">windowSpec	"Tools.UIPainter new openOnClass: self andSelector: #windowSpec"	&lt;resource: #canvas&gt;	^#(#{UI.FullSpec} 		#window: 		#(#{UI.WindowSpec} 			#properties: #(#{UI.PropertyListDictionary} #sizeType #specifiedSize #positionType #lastSavedPosition #openType #advanced #positionAutoSave true ) 			#label: 'StMPI Image Processing' 			#min: #(#{Core.Point} 20 20 ) 			#max: #(#{Core.Point} 0 0 ) 			#bounds: #(#{Graphics.Rectangle} 216 359 801 704 ) ) 		#component: 		#(#{UI.SpecCollection} 			#collection: #(				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 544 0 79 0 ) 							#name: #ImageOperation 							#flags: 16 							#label: 'Image Operation' ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 17 0 33 0 91 0 54 0 ) 							#name: #ApplySobel 							#flags: 24 							#model: #ApplySobel 							#label: 'Apply Sobel' 							#defaultable: true ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 268 0 33 0 394 0 54 0 ) 							#name: #DecreaseBrightness 							#flags: 24 							#model: #DecreaseBrightness 							#label: 'Decrease Brightness' 							#defaultable: true ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 119 0 33 0 239 0 54 0 ) 							#name: #IncreaseBrightness 							#flags: 24 							#model: #IncreaseBrightness 							#label: 'Increase Brightness' 							#defaultable: true ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 16 0 33 0 90 0 54 0 ) 							#name: #UndoSobel 							#flags: 24 							#model: #UndoSobel 							#label: 'Undo Sobel' 							#defaultable: true ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 21 0 169 0 565 0 248 0 ) 						#name: #ImageOperationGroup ) ) 				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 546 0 63 0 ) 							#name: #Result 							#flags: 16 							#label: 'Result' ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 18 0 31 0 ) 							#name: #ProcessingTimeLabel 							#flags: 16 							#label: 'Processing Time (s):' ) 						#(#{UI.InputFieldSpec} 							#layout: #(#{Graphics.LayoutFrame} 137 0 29 0 292 0 48 0 ) 							#name: #ProcessingTime 							#flags: 24 							#model: #ProcessingTime 							#type: #fixedpoint ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 20 0 262 0 566 0 325 0 ) 						#name: #ResultGroup ) ) 				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 545 0 66 0 ) 							#name: #Operation 							#flags: 16 							#label: 'Operation' ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 155 0 28 0 229 0 49 0 ) 							#name: #SaveImage 							#flags: 24 							#model: #SaveImage 							#label: 'Save Image' 							#defaultable: true ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 26 0 27 0 130 0 48 0 ) 							#name: #CloseImage 							#flags: 24 							#model: #CloseImage 							#label: 'Close Image' 							#defaultable: true ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 23 0 27 0 94 0 48 0 ) 							#name: #LoadImage 							#flags: 24 							#model: #LoadImage 							#label: 'Load Image' 							#defaultable: true ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 20 0 88 0 565 0 154 0 ) 						#name: #OperationGroup ) ) 				#(#{UI.CompositeSpecCollection} 					#collection: #(						#(#{UI.GroupBoxSpec} 							#layout: #(#{Graphics.LayoutFrame} 0 0 0 0 541 0 70 0 ) 							#name: #MPIControl 							#label: 'MPI Control' ) 						#(#{UI.SpinButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 175 0 28 0 275 0 49 0 ) 							#name: #numproc 							#model: #numproc 							#tabable: true 							#alignment: #center 							#type: #number 							#low: 2 							#interval: 1 ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 297 0 28 0 ) 							#name: #ParallelTechnique 							#label: 'Choose Parallel Technique :' ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 447 0 14 0 521 0 35 0 ) 							#name: #UseCudaButton 							#model: #UseCudaButton 							#label: 'Cuda' 							#defaultable: true ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 447 0 42 0 521 0 63 0 ) 							#name: #UseMPIButton 							#model: #UseMPIButton 							#label: 'MPI' 							#defaultable: true ) 						#(#{UI.LabelSpec} 							#layout: #(#{Graphics.LayoutOrigin} 14 0 30 0 ) 							#name: #numprocLabel 							#label: 'Number of processor to use :' ) 						#(#{UI.ActionButtonSpec} 							#layout: #(#{Graphics.LayoutFrame} 17 0 29 0 158 0 50 0 ) 							#name: #ChangeParallelTechnique 							#flags: 24 							#model: #ChangeParallelTechnique 							#label: 'Change Parallel Setting' 							#defaultable: true ) ) 					#compositeSpec: 					#(#{UI.CompositeSpec} 						#layout: #(#{Graphics.LayoutFrame} 23 0 9 0 564 0 79 0 ) 						#name: #MPIControlGroup ) ) 				#(#{UI.LabelSpec} 					#layout: #(#{Graphics.LayoutOrigin} 40 0 294 0 ) 					#name: #Wait 					#flags: 16 					#label: 'Please wait. Image is still in process.' ) ) ) )</body>
</methods>

<!-- -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -->


<methods>
<class-id>ImageProcessing</class-id> <category>actions</category>

<body package="ImageProcessingApp" selector="ApplySobel">ApplySobel	"This stub method was generated by UIDefiner "	| flagPtr requestPtr wrapper totalTime |	(self builder componentAt: #Wait) beVisible.	(self builder componentAt: #Result) beVisible.	(self builder componentAt: #ProcessingTime) beInvisible.	(self builder componentAt: #ProcessingTimeLabel) beInvisible.	PT == 1		ifTrue: 			[Dialog warn: ('This process will use &lt;1p&gt; core. Click OK to continue.'						expandMacrosWith: self numproc value)].	requestPtr := CIntegerType char gcCalloc.	flagPtr := CIntegerType int gcCalloc.	totalTime := CLimitedPrecisionRealType double gcCalloc.	flagPtr contents: 2.	StMPI new		MPI_Ibcast: flagPtr		with: 1		with: StMPI new MPI_INT		with: StMPI new MPI_ROOT		with: intercommSobelPtr contents		with: requestPtr.	StMPI new MPI_Wait: requestPtr with: StMPI new MPI_STATUS_IGNORE.	StMPI new		MPI_Recv: totalTime		with: 1		with: StMPI new MPI_DOUBLE		with: 0		with: 0		with: intercommSobelPtr contents		with: StMPI new MPI_STATUS_IGNORE.	self ProcessingTime value: totalTime contents.	wrapper := self wrapperAt: #Result.	wrapper beVisible.	wrapper := self wrapperAt: #UndoSobel.	wrapper beVisible.	wrapper := self wrapperAt: #ApplySobel.	wrapper beInvisible.	wrapper := self wrapperAt: #ProcessingTime.	wrapper beVisible.	wrapper := self wrapperAt: #ProcessingTimeLabel.	wrapper beVisible.	(self builder componentAt: #Wait) beInvisible.	^self</body>

<body package="ImageProcessingApp" selector="ChangeParallelTechnique">ChangeParallelTechnique	| wrapper |	wrapper := self wrapperAt: #OperationGroup.	wrapper beInvisible.	wrapper := self wrapperAt: #ParallelTechnique.	wrapper beVisible.	wrapper := self wrapperAt: #UseCudaButton.	wrapper beVisible.	wrapper := self wrapperAt: #UseMPIButton.	wrapper beVisible.	wrapper := self wrapperAt: #ChangeParallelTechnique.	wrapper beInvisible.	wrapper := self wrapperAt: #ImageOperationGroup.	wrapper beInvisible.	wrapper := self wrapperAt: #numproc.	wrapper beVisible.	wrapper := self wrapperAt: #numprocLabel.	wrapper beVisible.	intercommSobelPtr ~= nil		ifTrue: 			[self CloseImage].	^self</body>

<body package="ImageProcessingApp" selector="CloseImage">CloseImage	| flagPtr requestPtr wrapper |	requestPtr := CIntegerType char gcCalloc.	flagPtr := CIntegerType int gcCalloc.	flagPtr contents: 1.	StMPI new		MPI_Ibcast: flagPtr		with: 1		with: StMPI new MPI_INT		with: StMPI new MPI_ROOT		with: intercommSobelPtr contents		with: requestPtr.	StMPI new MPI_Wait: requestPtr with: StMPI new MPI_STATUS_IGNORE.	StMPI new MPI_Comm_disconnect: intercommSobelPtr.		intercommSobelPtr free.	intercommSobelPtr := nil.	requestPtr free.	flagPtr free.	wrapper := self wrapperAt: #CloseImage.	wrapper beInvisible.	wrapper := self wrapperAt: #LoadImage.	wrapper beVisible.	wrapper := self wrapperAt: #ApplySobel.	wrapper beInvisible.	wrapper := self wrapperAt: #ImageOperation.	wrapper beInvisible.	wrapper := self wrapperAt: #IncreaseBrightness.	wrapper beInvisible.	wrapper := self wrapperAt: #DecreaseBrightness.	wrapper beInvisible.	wrapper := self wrapperAt: #UndoSobel.	wrapper beInvisible.	wrapper := self wrapperAt: #SaveImage.	wrapper beInvisible.	wrapper := self wrapperAt: #Result.	wrapper beInvisible.	wrapper := self wrapperAt: #ProcessingTime.	wrapper beInvisible.	wrapper := self wrapperAt: #ProcessingTimeLabel.	wrapper beInvisible.	^self</body>

<body package="ImageProcessingApp" selector="DecreaseBrightness">DecreaseBrightness	| flagPtr requestPtr wrapper |	requestPtr := CIntegerType char gcCalloc.	flagPtr := CIntegerType int gcCalloc.	flagPtr contents: 5.	StMPI new		MPI_Ibcast: flagPtr		with: 1		with: StMPI new MPI_INT		with: StMPI new MPI_ROOT		with: intercommSobelPtr contents		with: requestPtr.	StMPI new MPI_Wait: requestPtr with: StMPI new MPI_STATUS_IGNORE.	^self</body>

<body package="ImageProcessingApp" selector="IncreaseBrightness">IncreaseBrightness	| flagPtr requestPtr wrapper |	requestPtr := CIntegerType char gcCalloc.	flagPtr := CIntegerType int gcCalloc.	flagPtr contents: 4.	StMPI new		MPI_Ibcast: flagPtr		with: 1		with: StMPI new MPI_INT		with: StMPI new MPI_ROOT		with: intercommSobelPtr contents		with: requestPtr.	StMPI new MPI_Wait: requestPtr with: StMPI new MPI_STATUS_IGNORE.	^self</body>

<body package="ImageProcessingApp" selector="LoadImage">LoadImage	| errcodePtr sendMessage spawnCount wrapper spawnName |	intercommSobelPtr := CIntegerType int gcCalloc.	sendMessage := OpenFileDialog new.	sendMessage addFileFilter: 'PGM file (*.pgm)' pattern: '*.pgm'.	sendMessage select.	sendMessage := sendMessage selectionString.	sendMessage ~= ''		ifTrue: 			[PT == 0				ifTrue: 					[spawnCount := 1.					spawnName := 'SobelFilterCUDA'].			PT == 1				ifTrue: 					[spawnCount := self numproc value.					spawnName := 'SobelFilterMPI'.					wrapper := self wrapperAt: #SaveImage.					wrapper beVisible].			errcodePtr := CIntegerType int pointerType gcMalloc: spawnCount.			StMPI new				MPI_Comm_spawn: spawnName				with: StMPI new MPI_ARGV_NULL				with: spawnCount				with: StMPI new MPI_INFO_NULL				with: 0				with: StMPI new MPI_COMM_SELF				with: intercommSobelPtr				with: errcodePtr.			StMPI new				MPI_Bcast: sendMessage				with: 256				with: StMPI new MPI_CHAR				with: StMPI new MPI_ROOT				with: intercommSobelPtr contents.			errcodePtr free.			wrapper := self wrapperAt: #CloseImage.			wrapper beVisible.			wrapper := self wrapperAt: #ImageOperationGroup.			wrapper beVisible.			wrapper := self wrapperAt: #ApplySobel.			wrapper beVisible.			wrapper := self wrapperAt: #UndoSobel.			wrapper beInvisible.			wrapper := self wrapperAt: #ImageOperation.			wrapper beVisible.			wrapper := self wrapperAt: #IncreaseBrightness.			wrapper beVisible.			wrapper := self wrapperAt: #DecreaseBrightness.			wrapper beVisible.			wrapper := self wrapperAt: #numproc.			wrapper beInvisible.			wrapper := self wrapperAt: #numprocLabel.			wrapper beInvisible.			wrapper := self wrapperAt: #LoadImage.			wrapper beInvisible].	^self</body>

<body package="ImageProcessingApp" selector="SaveImage">SaveImage	| flagPtr requestPtr |	requestPtr := CIntegerType char gcCalloc.	flagPtr := CIntegerType int gcCalloc.	flagPtr contents: 6.	StMPI new		MPI_Ibcast: flagPtr		with: 1		with: StMPI new MPI_INT		with: StMPI new MPI_ROOT		with: intercommSobelPtr contents		with: requestPtr.	StMPI new MPI_Wait: requestPtr with: StMPI new MPI_STATUS_IGNORE.	Dialog warn: 'Image was successfully saved. Click OK to continue'.	^self</body>

<body package="ImageProcessingApp" selector="UndoSobel">UndoSobel	| flagPtr requestPtr wrapper |	requestPtr := CIntegerType char gcCalloc.	flagPtr := CIntegerType int gcCalloc.	flagPtr contents: 3.	StMPI new		MPI_Ibcast: flagPtr		with: 1		with: StMPI new MPI_INT		with: StMPI new MPI_ROOT		with: intercommSobelPtr contents		with: requestPtr.	StMPI new MPI_Wait: requestPtr with: StMPI new MPI_STATUS_IGNORE.	wrapper := self wrapperAt: #ApplySobel.	wrapper beVisible.	wrapper := self wrapperAt: #UndoSobel.	wrapper beInvisible.	^self</body>

<body package="ImageProcessingApp" selector="UseCudaButton">UseCudaButton| wrapper |	PT := 0. "Cuda"	wrapper := self wrapperAt: #OperationGroup.	wrapper beVisible.	wrapper := self wrapperAt: #CloseImage.	wrapper beInvisible.	wrapper := self wrapperAt: #SaveImage.	wrapper beInvisible.	wrapper := self wrapperAt: #ChangeParallelTechnique.	wrapper beVisible.	wrapper := self wrapperAt: #UseCudaButton.	wrapper beInvisible.	wrapper := self wrapperAt: #UseMPIButton.	wrapper beInvisible.	wrapper := self wrapperAt: #numproc.	wrapper beInvisible.	wrapper := self wrapperAt: #numprocLabel.	wrapper beInvisible.	wrapper := self wrapperAt: #ParallelTechnique.	wrapper beInvisible.				^self</body>

<body package="ImageProcessingApp" selector="UseMPIButton">UseMPIButton| wrapper |	PT := 1.  "StMPI"	wrapper := self wrapperAt: #OperationGroup.	wrapper beVisible.	wrapper := self wrapperAt: #CloseImage.	wrapper beInvisible.	wrapper := self wrapperAt: #SaveImage.	wrapper beInvisible.	wrapper := self wrapperAt: #ChangeParallelTechnique.	wrapper beVisible.	wrapper := self wrapperAt: #UseCudaButton.	wrapper beInvisible.	wrapper := self wrapperAt: #UseMPIButton.	wrapper beInvisible.	wrapper := self wrapperAt: #numproc.	wrapper beInvisible.	wrapper := self wrapperAt: #numprocLabel.	wrapper beInvisible.	wrapper := self wrapperAt: #ParallelTechnique.	wrapper beInvisible.	^self</body>
</methods>

<methods>
<class-id>ImageProcessing</class-id> <category>aspects</category>

<body package="ImageProcessingApp" selector="ProcessingTime">ProcessingTime	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^ProcessingTime isNil		ifTrue:			[ProcessingTime := 0.00s asValue]		ifFalse:			[ProcessingTime]</body>

<body package="ImageProcessingApp" selector="numproc">numproc	"This method was generated by UIDefiner.  Any edits made here	may be lost whenever methods are automatically defined.  The	initialization provided below may have been preempted by an	initialize method."	^numproc isNil		ifTrue:			[numproc := 2 asValue]		ifFalse:			[numproc]</body>
</methods>

<methods>
<class-id>ImageProcessing</class-id> <category>initialize-release</category>

<body package="ImageProcessingApp" selector="initialize">initialize| commandLine argcPtr argvPtrPtr tempPtr argvPtrPtrPtr|	Transcript show: ('&lt;n&gt;&lt;n&gt;&lt;1p&gt;'				expandMacrosWith: 'StMPI Image Processing.').	commandLine := CEnvironment commandLine.	argcPtr := CIntegerType int gcCalloc.	argcPtr contents: commandLine size.	argvPtrPtr := CIntegerType char pointerType gcMalloc: commandLine size.	tempPtr := argvPtrPtr copy.	commandLine do: 			[:arg |			tempPtr contents: arg gcCopyToHeap.			tempPtr += 1].	argvPtrPtrPtr := CIntegerType char pointerType pointerType gcMalloc.	argvPtrPtrPtr contents: argvPtrPtr.	StMPI new MPI_Init: argcPtr with: argvPtrPtrPtr.	argcPtr free.	argvPtrPtr free.	argvPtrPtrPtr free.	^self</body>
</methods>

<methods>
<class-id>ImageProcessing</class-id> <category>changing</category>

<body package="ImageProcessingApp" selector="changeRequest">changeRequestStMPI new MPI_Abort: StMPI new MPI_COMM_WORLD with: 1.	^self</body>
</methods>

</st-source>
