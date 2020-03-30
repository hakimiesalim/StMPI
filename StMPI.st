<?xml version="1.0"?>

<st-source>
<time-stamp>From VisualWorks® Personal Use Edition, 8.3 of July 28, 2017 on February 17, 2020 at 11:15:40 PM</time-stamp>


<class>
<name>StMPI</name>
<environment>Smalltalk</environment>
<super>External.ExternalInterface</super>
<private>false</private>
<indexed-type>none</indexed-type>
<inst-vars></inst-vars>
<class-inst-vars></class-inst-vars>
<imports>
			private StMPIDictionary.*
			</imports>
<category>ExternalInterface-New</category>
<attributes>
<includeFiles><item>mpi.h</item></includeFiles>
<includeDirectories><item>C:\Program Files (x86)\Microsoft SDKs\MPI\Include</item></includeDirectories>
<libraryFiles><item>msmpi.dll</item></libraryFiles>
<libraryDirectories><item>C:\Windows\System32</item></libraryDirectories>
<beVirtual>false</beVirtual>
<optimizationLevel>full</optimizationLevel>
<package>StMPI GUI</package>
</attributes>
</class>

<methods>
<class-id>StMPI</class-id> <category>types</category>

<body package="StMPI GUI" selector="MPI_Aint">MPI_Aint	&lt;C: typedef int MPI_Aint&gt;</body>

<body package="StMPI GUI" selector="MPI_Comm">MPI_Comm	&lt;C: typedef int MPI_Comm&gt;</body>

<body package="StMPI GUI" selector="MPI_Comm_copy_attr_function">MPI_Comm_copy_attr_function	&lt;C: typedef int __stdcall  MPI_Comm_copy_attr_function(MPI_Comm oldcomm, int comm_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag)&gt;</body>

<body package="StMPI GUI" selector="MPI_Comm_delete_attr_function">MPI_Comm_delete_attr_function	&lt;C: typedef int __stdcall  MPI_Comm_delete_attr_function(MPI_Comm comm, int comm_keyval, void * attribute_val, void * extra_state)&gt;</body>

<body package="StMPI GUI" selector="MPI_Comm_errhandler_fn">MPI_Comm_errhandler_fn	&lt;C: typedef void __stdcall  MPI_Comm_errhandler_fn(MPI_Comm * comm, int * errcode, ...)&gt;</body>

<body package="StMPI GUI" selector="MPI_Count">MPI_Count	&lt;C: typedef int64_t MPI_Count&gt;</body>

<body package="StMPI GUI" selector="MPI_Datarep_conversion_function">MPI_Datarep_conversion_function	&lt;C: typedef int __stdcall  MPI_Datarep_conversion_function(void * userbuf, MPI_Datatype datatype, int count, void * filebuf, MPI_Offset position, void * extra_state)&gt;</body>

<body package="StMPI GUI" selector="MPI_Datarep_extent_function">MPI_Datarep_extent_function	&lt;C: typedef int __stdcall  MPI_Datarep_extent_function(MPI_Datatype datatype, MPI_Aint * file_extent, void * extra_state)&gt;</body>

<body package="StMPI GUI" selector="MPI_Datatype">MPI_Datatype	&lt;C: typedef int MPI_Datatype&gt;</body>

<body package="StMPI GUI" selector="MPI_Errhandler">MPI_Errhandler	&lt;C: typedef int MPI_Errhandler&gt;</body>

<body package="StMPI GUI" selector="MPI_File">MPI_File	&lt;C: typedef struct ADIOI_FileD * MPI_File&gt;</body>

<body package="StMPI GUI" selector="MPI_File_errhandler_fn">MPI_File_errhandler_fn	&lt;C: typedef void __stdcall  MPI_File_errhandler_fn(MPI_File * file, int * errcode, ...)&gt;</body>

<body package="StMPI GUI" selector="MPI_Fint">MPI_Fint	&lt;C: typedef int MPI_Fint&gt;</body>

<body package="StMPI GUI" selector="MPI_Grequest_cancel_function">MPI_Grequest_cancel_function	&lt;C: typedef int __stdcall  MPI_Grequest_cancel_function(void * extra_state, int complete)&gt;</body>

<body package="StMPI GUI" selector="MPI_Grequest_free_function">MPI_Grequest_free_function	&lt;C: typedef int __stdcall  MPI_Grequest_free_function(void * extra_state)&gt;</body>

<body package="StMPI GUI" selector="MPI_Grequest_query_function">MPI_Grequest_query_function	&lt;C: typedef int __stdcall  MPI_Grequest_query_function(void * extra_state, MPI_Status * status)&gt;</body>

<body package="StMPI GUI" selector="MPI_Group">MPI_Group	&lt;C: typedef int MPI_Group&gt;</body>

<body package="StMPI GUI" selector="MPI_Info">MPI_Info	&lt;C: typedef int MPI_Info&gt;</body>

<body package="StMPI GUI" selector="MPI_Message">MPI_Message	&lt;C: typedef int MPI_Message&gt;</body>

<body package="StMPI GUI" selector="MPI_Offset">MPI_Offset	&lt;C: typedef int64_t MPI_Offset&gt;</body>

<body package="StMPI GUI" selector="MPI_Op">MPI_Op	&lt;C: typedef int MPI_Op&gt;</body>

<body package="StMPI GUI" selector="MPI_Request">MPI_Request	&lt;C: typedef int MPI_Request&gt;</body>

<body package="StMPI GUI" selector="MPI_Status">MPI_Status	&lt;C: typedef struct __MPI_Status MPI_Status&gt;</body>

<body package="StMPI GUI" selector="MPI_Type_copy_attr_function">MPI_Type_copy_attr_function	&lt;C: typedef int __stdcall  MPI_Type_copy_attr_function(MPI_Datatype olddatatype, int datatype_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag)&gt;</body>

<body package="StMPI GUI" selector="MPI_Type_delete_attr_function">MPI_Type_delete_attr_function	&lt;C: typedef int __stdcall  MPI_Type_delete_attr_function(MPI_Datatype datatype, int datatype_keyval, void * attribute_val, void * extra_state)&gt;</body>

<body package="StMPI GUI" selector="MPI_User_function">MPI_User_function	&lt;C: typedef void __stdcall  MPI_User_function(void * invec, void * inoutvec, int * len, MPI_Datatype * datatype)&gt;</body>

<body package="StMPI GUI" selector="MPI_Win">MPI_Win	&lt;C: typedef int MPI_Win&gt;</body>

<body package="StMPI GUI" selector="MPI_Win_copy_attr_function">MPI_Win_copy_attr_function	&lt;C: typedef int __stdcall  MPI_Win_copy_attr_function(MPI_Win oldwin, int win_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag)&gt;</body>

<body package="StMPI GUI" selector="MPI_Win_delete_attr_function">MPI_Win_delete_attr_function	&lt;C: typedef int __stdcall  MPI_Win_delete_attr_function(MPI_Win win, int win_keyval, void * attribute_val, void * extra_state)&gt;</body>

<body package="StMPI GUI" selector="MPI_Win_errhandler_fn">MPI_Win_errhandler_fn	&lt;C: typedef void __stdcall  MPI_Win_errhandler_fn(MPI_Win * win, int * errcode, ...)&gt;</body>

<body package="StMPI GUI" selector="__declspec">__declspec	&lt;C: typedef void * __declspec&gt;</body>

<body package="StMPI GUI" selector="deprecated">deprecated	&lt;C: typedef void * deprecated&gt;</body>

<body package="StMPI GUI" selector="int64_t">int64_t	&lt;C: typedef void * int64_t&gt;</body>
</methods>

<methods>
<class-id>StMPI</class-id> <category>structs</category>

<body package="StMPI GUI" selector="ADIOI_FileD">ADIOI_FileD	&lt;C: struct ADIOI_FileD&gt;</body>

<body package="StMPI GUI" selector="__MPI_Status">__MPI_Status	&lt;C: struct __MPI_Status {			int internal[2];			int MPI_SOURCE, MPI_TAG, MPI_ERROR;		}&gt;</body>
</methods>

<methods>
<class-id>StMPI</class-id> <category>example</category>

<body package="StMPI GUI" selector="FloatArray">FloatArray &lt;C: typedef float FloatArray[10]&gt;</body>

<body package="StMPI GUI" selector="IntArray">IntArray&lt;C: typedef int IntArray[10]&gt;</body>
</methods>

<methods>
<class-id>StMPI</class-id> <category>example</category>

<body package="StMPI GUI" selector="FIRST">FIRST    "#define FIRST = 0"    ^0</body>

<body package="StMPI GUI" selector="LIMIT">LIMIT    "#define LIMIT = 10000000"    ^100000000</body>

<body package="StMPI GUI" selector="exampleArray">exampleArray	"https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_array.c"	| commandLine argcPtr argvPtrPtr argvPtrPtrPtr tempPtr taskidPtr numtasksPtr data chunksize ARRAYSIZE leftover tag1 tag2 MASTER sumPtr i offsetPtr dest mysumPtr source j |	MASTER := 0.	ARRAYSIZE := 20000.	data := CLimitedPrecisionRealType double gcMalloc: ARRAYSIZE.	sumPtr := CLimitedPrecisionRealType double gcCalloc.	offsetPtr := CIntegerType int gcCalloc.	mysumPtr := CLimitedPrecisionRealType double gcCalloc.	Transcript		show: ('&lt;n&gt;&lt;n&gt;&lt;1p&gt;' expandMacrosWith: 'StMPIArray new exampleArray.').	commandLine := CEnvironment commandLine.	argcPtr := CIntegerType int gcCalloc.	argcPtr contents: commandLine size.	argvPtrPtr := CIntegerType char pointerType gcMalloc: commandLine size.	tempPtr := argvPtrPtr copy.	commandLine do: 			[:arg |			tempPtr contents: arg gcCopyToHeap.			tempPtr += 1].	argvPtrPtrPtr := CIntegerType char pointerType pointerType gcMalloc.	argvPtrPtrPtr contents: argvPtrPtr.	self MPI_Init: argcPtr with: argvPtrPtrPtr.	taskidPtr := CIntegerType int gcCalloc.	self MPI_Comm_rank: self MPI_COMM_WORLD with: taskidPtr.	numtasksPtr := CIntegerType int gcCalloc.	self MPI_Comm_size: self MPI_COMM_WORLD with: numtasksPtr.	Transcript		show: ('&lt;n&gt;MPI task &lt;1p&gt; has started' expandMacrosWith: taskidPtr contents).	chunksize := ARRAYSIZE // numtasksPtr contents.	leftover := ARRAYSIZE \\ numtasksPtr contents.	tag2 := 1.	tag1 := 2.	"/********Master task only*******/"	taskidPtr contents = MASTER		ifTrue: 			[sumPtr contents: 0.			i := 0.			[i &lt; ARRAYSIZE] whileTrue: 					[data at: i put: i.					sumPtr contents: sumPtr contents + (data at: i).					i := i + 1].			Transcript				show: ('&lt;n&gt;Initialized array sum = &lt;1p&gt;' expandMacrosWith: sumPtr contents).			Transcript show: ('&lt;n&gt; numtasks = &lt;1p&gt;		chunksize = &lt;2p&gt;		leftover = &lt;3p&gt;'						expandMacrosWith: numtasksPtr contents						with: chunksize						with: leftover).			"**Send each task its portion of the array - master keeps 1st part plus leftover elements**"			offsetPtr contents: chunksize + leftover.			dest := 1.			[dest &lt; numtasksPtr contents] whileTrue: 					[self						MPI_Send: offsetPtr						with: 1						with: self MPI_INT						with: dest						with: tag1						with: self MPI_COMM_WORLD.					self						MPI_Send: (data + offsetPtr contents)						with: chunksize						with: self MPI_DOUBLE						with: dest						with: tag2						with: self MPI_COMM_WORLD.					Transcript show: ('&lt;n&gt;Sent &lt;1p&gt; elements to task &lt;2p&gt; offset = &lt;3p&gt;'								expandMacrosWith: chunksize								with: dest								with: offsetPtr contents).					offsetPtr contents: offsetPtr contents + chunksize.					dest := dest + 1].			"**Master does its part of the work**"			offsetPtr contents: 0.			mysumPtr contents: (self						update: offsetPtr contents						with: chunksize + leftover						with: taskidPtr contents						with: data).			"**Wait to receive results from each task**"			i := 1.			[i &lt; numtasksPtr contents] whileTrue: 					[source := 1.					self						MPI_Recv: offsetPtr						with: 1						with: self MPI_INT						with: source						with: tag1						with: self MPI_COMM_WORLD						with: self MPI_STATUS_IGNORE.					self						MPI_Recv: (data + offsetPtr contents)						with: chunksize						with: self MPI_DOUBLE						with: source						with: tag2						with: self MPI_COMM_WORLD						with: self MPI_STATUS_IGNORE.					i := i + 1].			"**Get final sum and print sample results**"			self				MPI_Reduce: mysumPtr				with: sumPtr				with: 1				with: self MPI_DOUBLE				with: self MPI_SUM				with: MASTER				with: self MPI_COMM_WORLD.			Transcript show: '&lt;n&gt;Sample results: &lt;n&gt;'.			offsetPtr contents: 0.			i := 0.			[i &lt; numtasksPtr contents] whileTrue: 					[j := 0.					[j &lt; 5] whileTrue: 							[Transcript								show: ('     &lt;1p&gt;' expandMacrosWith: (data at: offsetPtr contents + j)).							j := j + 1].					Transcript show: '&lt;n&gt;'.					offsetPtr contents: offsetPtr contents + chunksize.					i := i + 1].			Transcript				show: ('&lt;n&gt;*****Final sum = &lt;1p&gt; *****' expandMacrosWith: sumPtr contents)].	"**Initialize the array**"	"**End of master section**"	"***** Non-master tasks only*****"	taskidPtr contents &gt; MASTER		ifTrue: 			["**Receive my portion of array from the master task**"			source := MASTER.			self				MPI_Recv: offsetPtr				with: 1				with: self MPI_INT				with: source				with: tag1				with: self MPI_COMM_WORLD				with: self MPI_STATUS_IGNORE.			self				MPI_Recv: (data + offsetPtr contents)				with: chunksize				with: self MPI_DOUBLE				with: source				with: tag2				with: self MPI_COMM_WORLD				with: self MPI_STATUS_IGNORE.			"**Do my part of the work**"			mysumPtr contents: (self						update: offsetPtr contents						with: chunksize						with: taskidPtr contents						with: data).			"**Send my results back to the master task**"			dest := MASTER.			self				MPI_Send: offsetPtr				with: 1				with: self MPI_INT				with: dest				with: tag1				with: self MPI_COMM_WORLD.			self				MPI_Send: (data + offsetPtr contents)				with: chunksize				with: self MPI_DOUBLE				with: MASTER				with: tag2				with: self MPI_COMM_WORLD.			"**Use sum reduction operation to obtain final sum**"			self				MPI_Reduce: mysumPtr				with: sumPtr				with: 1				with: self MPI_DOUBLE				with: self MPI_SUM				with: MASTER				with: self MPI_COMM_WORLD].	"** end of non-master**"	self MPI_Finalize</body>

<body package="StMPI GUI" selector="exampleHelloWorldWikipedia">exampleHelloWorldWikipedia    "self new exampleHelloWorldWikipedia."    "https://en.wikipedia.org/wiki/Message_Passing_Interface#Example_program"    | commandLine argcPtr argvPtrPtr argvPtrPtrPtr tempPtr my_rankPtr num_procsPtr sendMessage charBufferSize charBufferPtr sendMessageSize recvMessage userInput|    Transcript show: ('&lt;n&gt;&lt;n&gt;&lt;1p&gt;'                expandMacrosWith: 'StMPIHelloWorld new exampleHelloWorldWikipedia.').    commandLine := CEnvironment commandLine.    argcPtr := CIntegerType int gcCalloc.    argcPtr contents: commandLine size.    argvPtrPtr := CIntegerType char pointerType gcMalloc: commandLine size.    tempPtr := argvPtrPtr copy.    commandLine do:            [:arg |            tempPtr contents: arg gcCopyToHeap.            tempPtr += 1].    argvPtrPtrPtr := CIntegerType char pointerType pointerType gcMalloc.    argvPtrPtrPtr contents: argvPtrPtr.    self MPI_Init: argcPtr with: argvPtrPtrPtr.    my_rankPtr := CIntegerType int gcCalloc.    self MPI_Comm_rank: self MPI_COMM_WORLD with: my_rankPtr.    num_procsPtr := CIntegerType int gcCalloc.    self MPI_Comm_size: self MPI_COMM_WORLD with: num_procsPtr.    charBufferSize := 256.    charBufferPtr := CIntegerType char gcMalloc: charBufferSize.	    [true]    whileTrue: [    my_rankPtr contents = 0        ifTrue:            [	Transcript                show: ('&lt;n&gt;We have &lt;1p&gt; processes.' expandMacrosWith: num_procsPtr contents).            Transcript                show: ('&lt;n&gt;My rank is &lt;1p&gt;.' expandMacrosWith: my_rankPtr contents).            1 to: num_procsPtr contents - 1                do:                    [:other_rank |                    sendMessage := 'Hello &lt;1p&gt;' expandMacrosWith: other_rank.                    sendMessageSize := sendMessage size min: charBufferSize - 1.                    0 to: sendMessageSize - 1                        do: [:i | charBufferPtr at: i put: (sendMessage at: i + 1) asInteger].                    charBufferPtr at: sendMessageSize + 1 put: 0.                    self                        MPI_Send: charBufferPtr                        with: charBufferSize                        with: self MPI_CHAR                        with: other_rank                        with: 0                        with: self MPI_COMM_WORLD].            1 to: num_procsPtr contents - 1                do:                    [:other_rank |                    self                        MPI_Recv: charBufferPtr                        with: charBufferSize                        with: self MPI_CHAR                        with: other_rank                        with: 0                        with: self MPI_COMM_WORLD                        with: self MPI_STATUS_IGNORE.                    recvMessage := charBufferPtr copyCStringFromHeap.                    Transcript show: ('&lt;n&gt;Recv: &lt;1s&gt;' expandMacrosWith: recvMessage)].	userInput := Dialog		choose: 'Do you want to repeat the process?'		labels: (Array with: 'Yes' with: 'No')		values: #(#yes #no)		default: #yes.	userInput == #no	ifTrue: [self MPI_Abort: self MPI_COMM_WORLD with: 1.].	]        ifFalse:            [Transcript                show: ('&lt;n&gt;My rank is &lt;1p&gt;.' expandMacrosWith: my_rankPtr contents).            self                MPI_Recv: charBufferPtr                with: charBufferSize                with: self MPI_CHAR                with: 0                with: 0                with: self MPI_COMM_WORLD                with: self MPI_STATUS_IGNORE.            recvMessage := charBufferPtr copyCStringFromHeap.            Transcript show: ('&lt;n&gt;Recv: &lt;1s&gt;' expandMacrosWith: recvMessage).            sendMessage := 'Process &lt;1p&gt; reporting for duty.'                        expandMacrosWith: my_rankPtr contents.            sendMessageSize := sendMessage size min: charBufferSize - 1.            0 to: sendMessageSize - 1                do: [:i | charBufferPtr at: i put: (sendMessage at: i + 1) asInteger].            charBufferPtr at: sendMessageSize + 1 put: 0.            self                MPI_Send: charBufferPtr                with: charBufferSize                with: self MPI_CHAR                with: 0                with: 0                with: self MPI_COMM_WORLD].	].</body>

<body package="StMPI GUI" selector="examplePrimeNumberWithSlave">examplePrimeNumberWithSlave"https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_prime.c"|commandLine argcPtr argvPtrPtr argvPtrPtrPtr tempPtr rankPtr ntasksPtr pcPtr n pcsumPtr foundonePtr maxprimePtr mystart stride start_time end_time limitPtr leftOverPtr|Transcript show: ('&lt;n&gt;&lt;n&gt;&lt;1p&gt;'                expandMacrosWith: 'StMPIPrimeNumber new examplePrimeNumber.').    commandLine := CEnvironment commandLine.    argcPtr := CIntegerType int gcCalloc.    argcPtr contents: commandLine size.    argvPtrPtr := CIntegerType char pointerType gcMalloc: commandLine size.    tempPtr := argvPtrPtr copy.    commandLine do:            [:arg |            tempPtr contents: arg gcCopyToHeap.            tempPtr += 1].    argvPtrPtrPtr := CIntegerType char pointerType pointerType gcMalloc.    argvPtrPtrPtr contents: argvPtrPtr.    self MPI_Init: argcPtr with: argvPtrPtrPtr.    rankPtr := CIntegerType int gcCalloc.    self MPI_Comm_rank: self MPI_COMM_WORLD with: rankPtr.    ntasksPtr := CIntegerType int gcCalloc.    self MPI_Comm_size: self MPI_COMM_WORLD with: ntasksPtr.leftOverPtr := CIntegerType int gcCalloc.leftOverPtr contents: 0.limitPtr := CIntegerType int gcCalloc.limitPtr contents: (self LIMIT).    [(limitPtr contents \\ ntasksPtr contents) ~= 0]	whileTrue: [limitPtr contents: ((limitPtr contents) - 1).	leftOverPtr contents: ((leftOverPtr contents) + 1).		].	start_time := self MPI_Wtime.	mystart := ((rankPtr contents * 2) + 1).	stride := (ntasksPtr contents * 2).	pcPtr := CIntegerType int gcCalloc.	pcPtr contents: 0.	foundonePtr := CIntegerType int gcCalloc.	foundonePtr contents: 0.	pcsumPtr := CIntegerType int gcCalloc.	maxprimePtr := CIntegerType int gcCalloc.		"****************** task with rank 0 does this part ******************"rankPtr contents = self FIRSTifTrue: [Transcript show: ('&lt;n&gt;Using &lt;1p&gt; tasks ' expandMacrosWith: ntasksPtr contents).	Transcript show: ('to scan &lt;1p&gt; numbers.' expandMacrosWith: ((limitPtr contents) + (leftOverPtr contents))).	pcPtr contents: 4.		n := 1.	[n &lt;= ((ntasksPtr contents) - 1)]	whileTrue:[		self MPI_Send: limitPtr			with: 1			with: self MPI_INT			with: n			with: 0			with: self MPI_COMM_WORLD.		n := (n + 1).].	n := mystart.	[n &lt;= (limitPtr contents)]	whileTrue:[((self isPrime: n) = 1)		ifTrue: [pcPtr contents: (pcPtr contents + 1).			foundonePtr contents: n.			"Optional Transcript show: ('&lt;n&gt;&lt;1p&gt;' expandMacrosWith: foundonePtr contents). "			].		n := n + stride].			n := (limitPtr contents).	[n &lt;= (self LIMIT)]	whileTrue:[((self isPrime: n) = 1)		ifTrue: [pcPtr contents: (pcPtr contents + 1).			foundonePtr contents: n.			"Optional Transcript show: ('&lt;n&gt;&lt;1p&gt;' expandMacrosWith: foundonePtr contents). "			].		n := n + stride].	self MPI_Reduce: pcPtr	     with: pcsumPtr	     with: 1	     with: self MPI_INT	     with: self MPI_SUM	     with: self FIRST	     with: self MPI_COMM_WORLD.	self MPI_Reduce: foundonePtr	     with: maxprimePtr	     with: 1	     with: self MPI_INT	     with: self MPI_MAX	     with: self FIRST	     with: self MPI_COMM_WORLD.	end_time := self MPI_Wtime.	Transcript show:('&lt;n&gt;Done. Largest prime is &lt;1p&gt; ' expandMacrosWith: maxprimePtr contents).	Transcript show:('Total primes &lt;1p&gt;' expandMacrosWith: pcsumPtr contents).	Transcript show:('&lt;n&gt;Wallclock time elapsed: &lt;1p&gt; seconds' expandMacrosWith: (end_time - start_time)).]."****************** all other tasks do this part ******************"rankPtr contents &gt; self FIRSTifTrue: [	self MPI_Recv: limitPtr		with: 1		with: self MPI_INT		with: 0		with: 0		with: self MPI_COMM_WORLD		with: self MPI_STATUS_IGNORE.	n := mystart.	[n &lt;= (limitPtr contents)]	whileTrue:[((self isPrime: n) = 1)		ifTrue: [pcPtr contents: (pcPtr contents + 1).			foundonePtr contents: n.			"Optional Transcript show: ('&lt;n&gt;&lt;1p&gt;' expandMacrosWith: foundonePtr contents). "			].		n := n + stride].	self MPI_Reduce: pcPtr	     with: pcsumPtr	     with: 1	     with: self MPI_INT	     with: self MPI_SUM	     with: self FIRST	     with: self MPI_COMM_WORLD.	self MPI_Reduce: foundonePtr	     with: maxprimePtr	     with: 1	     with: self MPI_INT	     with: self MPI_MAX	     with: self FIRST	     with: self MPI_COMM_WORLD.	end_time := self MPI_Wtime.].	self MPI_Finalize</body>

<body package="StMPI GUI" selector="isPrime:">isPrime: n	| squareroot |	n &gt; 10	ifTrue: [squareroot := ((n sqrt) asInteger).		 3 to: squareroot by:2 do: [ :i | 			(n \\ i) = 0			ifTrue:[^0]].		^1]	ifFalse: [^0]</body>

<body package="StMPI GUI" selector="maxPrime">maxPrime &lt;C: extern int maxPrime&gt;</body>

<body package="StMPI GUI" selector="maxPrime:">maxPrime: value &lt;C: extern int maxPrime&gt;</body>

<body package="StMPI GUI" selector="update:with:with:">update: myoffset with: chunk with: myid 	| i mysum data|	mysum := 0.	data := self FloatArray malloc.	i := myoffset.	[i &lt; (myoffset + chunk)]	whileTrue:[data at: i put: ((data at: i) + (i * 1.0)).		mysum := (mysum + (data at:i)).		i := i + 1.].	Transcript show: ('&lt;n&gt;Task &lt;1p&gt; mysum = &lt;2p&gt;' expandMacrosWith: myid with: mysum).	 	^mysum</body>

<body package="StMPI GUI" selector="update:with:with:with:">update: myoffset with: chunk with: myid with: data	| i mysum |	mysum:= 0.	i := myoffset.	[i &lt; (myoffset + chunk)] whileTrue: 			[data at: i put: (data at: i) + i.			mysum := mysum + (data at: i).			i := i + 1].	Transcript		show: ('&lt;n&gt;Task &lt;1p&gt; mysum = &lt;2p&gt;' expandMacrosWith: myid with: mysum).	^mysum</body>
</methods>

<methods>
<class-id>StMPI</class-id> <category>procedures</category>

<body package="StMPI GUI" selector="MPI_Abort:with:">MPI_Abort: comm with: errorcode	&lt;C: int __stdcall  MPI_Abort(MPI_Comm comm, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Accumulate:with:with:with:with:with:with:with:with:">MPI_Accumulate: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win	&lt;C: int __stdcall  MPI_Accumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Add_error_class:">MPI_Add_error_class: errorclass	&lt;C: int __stdcall  MPI_Add_error_class(int * errorclass)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Add_error_code:with:">MPI_Add_error_code: errorclass with: errorcode	&lt;C: int __stdcall  MPI_Add_error_code(int errorclass, int * errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Add_error_string:with:">MPI_Add_error_string: errorcode with: string	&lt;C: int __stdcall  MPI_Add_error_string(int errorcode, const char * string)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Allgather:with:with:with:with:with:with:">MPI_Allgather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm	&lt;C: int __stdcall  MPI_Allgather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Allgatherv:with:with:with:with:with:with:with:">MPI_Allgatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: comm	&lt;C: int __stdcall  MPI_Allgatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Alloc_mem:with:with:">MPI_Alloc_mem: size with: info with: baseptr	&lt;C: int __stdcall  MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void * baseptr)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Allreduce:with:with:with:with:with:">MPI_Allreduce: sendbuf with: recvbuf with: count with: datatype with: op with: comm	&lt;C: int __stdcall  MPI_Allreduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Alltoall:with:with:with:with:with:with:">MPI_Alltoall: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm	&lt;C: int __stdcall  MPI_Alltoall(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Alltoallv:with:with:with:with:with:with:with:with:">MPI_Alltoallv: sendbuf with: sendcounts with: sdispls with: sendtype with: recvbuf with: recvcounts with: rdispls with: recvtype with: comm	&lt;C: int __stdcall  MPI_Alltoallv(const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Alltoallw:with:with:with:with:with:with:with:with:">MPI_Alltoallw: sendbuf with: sendcounts with: sdispls with: sendtypes with: recvbuf with: recvcounts with: rdispls with: recvtypes with: comm	&lt;C: int __stdcall  MPI_Alltoallw(const void * sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Barrier:">MPI_Barrier: comm	&lt;C: int __stdcall  MPI_Barrier(MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Bcast:with:with:with:with:">MPI_Bcast: buffer with: count with: datatype with: root with: comm	&lt;C: int __stdcall  MPI_Bcast(void * buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Bsend:with:with:with:with:with:">MPI_Bsend: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  MPI_Bsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Bsend_init:with:with:with:with:with:with:">MPI_Bsend_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Bsend_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Buffer_attach:with:">MPI_Buffer_attach: buffer with: size	&lt;C: int __stdcall  MPI_Buffer_attach(void * buffer, int size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Buffer_detach:with:">MPI_Buffer_detach: buffer_addr with: size	&lt;C: int __stdcall  MPI_Buffer_detach(void * buffer_addr, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cancel:">MPI_Cancel: request	&lt;C: int __stdcall  MPI_Cancel(MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_coords:with:with:with:">MPI_Cart_coords: comm with: rank with: maxdims with: coords	&lt;C: int __stdcall  MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_create:with:with:with:with:with:">MPI_Cart_create: comm_old with: ndims with: dims with: periods with: reorder with: comm_cart	&lt;C: int __stdcall  MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm * comm_cart)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_get:with:with:with:with:">MPI_Cart_get: comm with: maxdims with: dims with: periods with: coords	&lt;C: int __stdcall  MPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_map:with:with:with:with:">MPI_Cart_map: comm with: ndims with: dims with: periods with: newrank	&lt;C: int __stdcall  MPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int * newrank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_rank:with:with:">MPI_Cart_rank: comm with: coords with: rank	&lt;C: int __stdcall  MPI_Cart_rank(MPI_Comm comm, const int coords[], int * rank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_shift:with:with:with:with:">MPI_Cart_shift: comm with: direction with: disp with: rank_source with: rank_dest	&lt;C: int __stdcall  MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source, int * rank_dest)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cart_sub:with:with:">MPI_Cart_sub: comm with: remain_dims with: newcomm	&lt;C: int __stdcall  MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Cartdim_get:with:">MPI_Cartdim_get: comm with: ndims	&lt;C: int __stdcall  MPI_Cartdim_get(MPI_Comm comm, int * ndims)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Close_port:">MPI_Close_port: port_name	&lt;C: int __stdcall  MPI_Close_port(const char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_accept:with:with:with:with:">MPI_Comm_accept: port_name with: info with: root with: comm with: newcomm	&lt;C: int __stdcall  MPI_Comm_accept(const char * port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_call_errhandler:with:">MPI_Comm_call_errhandler: comm with: errorcode	&lt;C: int __stdcall  MPI_Comm_call_errhandler(MPI_Comm comm, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_compare:with:with:">MPI_Comm_compare: comm1 with: comm2 with: result	&lt;C: int __stdcall  MPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int * result)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_connect:with:with:with:with:">MPI_Comm_connect: port_name with: info with: root with: comm with: newcomm	&lt;C: int __stdcall  MPI_Comm_connect(const char * port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_create:with:with:">MPI_Comm_create: comm with: group with: newcomm	&lt;C: int __stdcall  MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_create_errhandler:with:">MPI_Comm_create_errhandler: function with: errhandler	&lt;C: int __stdcall  MPI_Comm_create_errhandler(void ( __stdcall * function)(MPI_Comm * comm, int * errcode, ...), MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_create_keyval:with:with:with:">MPI_Comm_create_keyval: comm_copy_attr_fn with: comm_delete_attr_fn with: comm_keyval with: extra_state	&lt;C: int __stdcall  MPI_Comm_create_keyval(int ( __stdcall * comm_copy_attr_fn)(MPI_Comm oldcomm, int comm_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag), int ( __stdcall * comm_delete_attr_fn)(MPI_Comm comm, int comm_keyval, void * attribute_val, void * extra_state), int * comm_keyval, void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_delete_attr:with:">MPI_Comm_delete_attr: comm with: comm_keyval	&lt;C: int __stdcall  MPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_disconnect:">MPI_Comm_disconnect: comm	&lt;C: int __stdcall  MPI_Comm_disconnect(MPI_Comm * comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_dup:with:">MPI_Comm_dup: comm with: newcomm	&lt;C: int __stdcall  MPI_Comm_dup(MPI_Comm comm, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_free:">MPI_Comm_free: comm	&lt;C: int __stdcall  MPI_Comm_free(MPI_Comm * comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_free_keyval:">MPI_Comm_free_keyval: comm_keyval	&lt;C: int __stdcall  MPI_Comm_free_keyval(int * comm_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_get_attr:with:with:with:">MPI_Comm_get_attr: comm with: comm_keyval with: attribute_val with: flag	&lt;C: int __stdcall  MPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void * attribute_val, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_get_errhandler:with:">MPI_Comm_get_errhandler: comm with: errhandler	&lt;C: int __stdcall  MPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_get_name:with:with:">MPI_Comm_get_name: comm with: comm_name with: resultlen	&lt;C: int __stdcall  MPI_Comm_get_name(MPI_Comm comm, char * comm_name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_get_parent:">MPI_Comm_get_parent: parent	&lt;C: int __stdcall  MPI_Comm_get_parent(MPI_Comm * parent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_group:with:">MPI_Comm_group: comm with: group	&lt;C: int __stdcall  MPI_Comm_group(MPI_Comm comm, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_join:with:">MPI_Comm_join: fd with: intercomm	&lt;C: int __stdcall  MPI_Comm_join(int fd, MPI_Comm * intercomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_rank:with:">MPI_Comm_rank: comm with: rank	&lt;C: int __stdcall  MPI_Comm_rank(MPI_Comm comm, int * rank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_remote_group:with:">MPI_Comm_remote_group: comm with: group	&lt;C: int __stdcall  MPI_Comm_remote_group(MPI_Comm comm, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_remote_size:with:">MPI_Comm_remote_size: comm with: size	&lt;C: int __stdcall  MPI_Comm_remote_size(MPI_Comm comm, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_set_attr:with:with:">MPI_Comm_set_attr: comm with: comm_keyval with: attribute_val	&lt;C: int __stdcall  MPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void * attribute_val)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_set_errhandler:with:">MPI_Comm_set_errhandler: comm with: errhandler	&lt;C: int __stdcall  MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_set_name:with:">MPI_Comm_set_name: comm with: comm_name	&lt;C: int __stdcall  MPI_Comm_set_name(MPI_Comm comm, const char * comm_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_size:with:">MPI_Comm_size: comm with: size	&lt;C: int __stdcall  MPI_Comm_size(MPI_Comm comm, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_spawn:with:with:with:with:with:with:with:">MPI_Comm_spawn: command with: argv with: maxprocs with: info with: root with: comm with: intercomm with: array_of_errcodes	&lt;C: int __stdcall  MPI_Comm_spawn(const char * command, char * argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * intercomm, int array_of_errcodes[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_spawn_multiple:with:with:with:with:with:with:with:with:">MPI_Comm_spawn_multiple: count with: array_of_commands with: array_of_argv with: array_of_maxprocs with: array_of_info with: root with: comm with: intercomm with: array_of_errcodes	&lt;C: int __stdcall  MPI_Comm_spawn_multiple(int count, char * array_of_commands[], char * * array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm * intercomm, int array_of_errcodes[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_split:with:with:with:">MPI_Comm_split: comm with: color with: key with: newcomm	&lt;C: int __stdcall  MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_split_type:with:with:with:with:">MPI_Comm_split_type: comm with: split_type with: key with: info with: newcomm	&lt;C: int __stdcall  MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Comm_test_inter:with:">MPI_Comm_test_inter: comm with: flag	&lt;C: int __stdcall  MPI_Comm_test_inter(MPI_Comm comm, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Compare_and_swap:with:with:with:with:with:with:">MPI_Compare_and_swap: origin_addr with: compare_addr with: result_addr with: datatype with: target_rank with: target_disp with: win	&lt;C: int __stdcall  MPI_Compare_and_swap(const void * origin_addr, const void * compare_addr, void * result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Dims_create:with:with:">MPI_Dims_create: nnodes with: ndims with: dims	&lt;C: int __stdcall  MPI_Dims_create(int nnodes, int ndims, int dims[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Dist_graph_create:with:with:with:with:with:with:with:with:">MPI_Dist_graph_create: comm_old with: n with: sources with: degrees with: destinations with: weights with: info with: reorder with: comm_dist_graph	&lt;C: int __stdcall  MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[], const int destinations[], const int weights[], MPI_Info info, int reorder, MPI_Comm * comm_dist_graph)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Dist_graph_create_adjacent:with:with:with:with:with:with:with:with:with:">MPI_Dist_graph_create_adjacent: comm_old with: indegree with: sources with: sourceweights with: outdegree with: destinations with: destweights with: info with: reorder with: comm_dist_graph	&lt;C: int __stdcall  MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm * comm_dist_graph)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Dist_graph_neighbors:with:with:with:with:with:with:">MPI_Dist_graph_neighbors: comm with: maxindegree with: sources with: sourceweights with: maxoutdegree with: destinations with: destweights	&lt;C: int __stdcall  MPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Dist_graph_neighbors_count:with:with:with:">MPI_Dist_graph_neighbors_count: comm with: indegree with: outdegree with: weighted	&lt;C: int __stdcall  MPI_Dist_graph_neighbors_count(MPI_Comm comm, int * indegree, int * outdegree, int * weighted)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Errhandler_free:">MPI_Errhandler_free: errhandler	&lt;C: int __stdcall  MPI_Errhandler_free(MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Error_class:with:">MPI_Error_class: errorcode with: errorclass	&lt;C: int __stdcall  MPI_Error_class(int errorcode, int * errorclass)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Error_string:with:with:">MPI_Error_string: errorcode with: string with: resultlen	&lt;C: int __stdcall  MPI_Error_string(int errorcode, char * string, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Exscan:with:with:with:with:with:">MPI_Exscan: sendbuf with: recvbuf with: count with: datatype with: op with: comm	&lt;C: int __stdcall  MPI_Exscan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Fetch_and_op:with:with:with:with:with:with:">MPI_Fetch_and_op: origin_addr with: result_addr with: datatype with: target_rank with: target_disp with: op with: win	&lt;C: int __stdcall  MPI_Fetch_and_op(const void * origin_addr, void * result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_call_errhandler:with:">MPI_File_call_errhandler: file with: errorcode	&lt;C: int __stdcall  MPI_File_call_errhandler(MPI_File file, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_close:">MPI_File_close: fh	&lt;C: int __stdcall  MPI_File_close(MPI_File * fh)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_create_errhandler:with:">MPI_File_create_errhandler: function with: errhandler	&lt;C: int __stdcall  MPI_File_create_errhandler(void ( __stdcall * function)(MPI_File * file, int * errcode, ...), MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_delete:with:">MPI_File_delete: filename with: info	&lt;C: int __stdcall  MPI_File_delete(const char * filename, MPI_Info info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_amode:with:">MPI_File_get_amode: fh with: amode	&lt;C: int __stdcall  MPI_File_get_amode(MPI_File fh, int * amode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_atomicity:with:">MPI_File_get_atomicity: fh with: flag	&lt;C: int __stdcall  MPI_File_get_atomicity(MPI_File fh, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_byte_offset:with:with:">MPI_File_get_byte_offset: fh with: offset with: disp	&lt;C: int __stdcall  MPI_File_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset * disp)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_errhandler:with:">MPI_File_get_errhandler: file with: errhandler	&lt;C: int __stdcall  MPI_File_get_errhandler(MPI_File file, MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_group:with:">MPI_File_get_group: fh with: group	&lt;C: int __stdcall  MPI_File_get_group(MPI_File fh, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_info:with:">MPI_File_get_info: fh with: info_used	&lt;C: int __stdcall  MPI_File_get_info(MPI_File fh, MPI_Info * info_used)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_position:with:">MPI_File_get_position: fh with: offset	&lt;C: int __stdcall  MPI_File_get_position(MPI_File fh, MPI_Offset * offset)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_position_shared:with:">MPI_File_get_position_shared: fh with: offset	&lt;C: int __stdcall  MPI_File_get_position_shared(MPI_File fh, MPI_Offset * offset)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_size:with:">MPI_File_get_size: fh with: size	&lt;C: int __stdcall  MPI_File_get_size(MPI_File fh, MPI_Offset * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_type_extent:with:with:">MPI_File_get_type_extent: fh with: datatype with: extent	&lt;C: int __stdcall  MPI_File_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint * extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_get_view:with:with:with:with:">MPI_File_get_view: fh with: disp with: etype with: filetype with: datarep	&lt;C: int __stdcall  MPI_File_get_view(MPI_File fh, MPI_Offset * disp, MPI_Datatype * etype, MPI_Datatype * filetype, char * datarep)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_iread:with:with:with:with:">MPI_File_iread: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  MPI_File_iread(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_iread_at:with:with:with:with:with:">MPI_File_iread_at: fh with: offset with: buf with: count with: datatype with: request	&lt;C: int __stdcall  MPI_File_iread_at(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_iread_shared:with:with:with:with:">MPI_File_iread_shared: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  MPI_File_iread_shared(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_iwrite:with:with:with:with:">MPI_File_iwrite: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  MPI_File_iwrite(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_iwrite_at:with:with:with:with:with:">MPI_File_iwrite_at: fh with: offset with: buf with: count with: datatype with: request	&lt;C: int __stdcall  MPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_iwrite_shared:with:with:with:with:">MPI_File_iwrite_shared: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  MPI_File_iwrite_shared(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_open:with:with:with:with:">MPI_File_open: comm with: filename with: amode with: info with: fh	&lt;C: int __stdcall  MPI_File_open(MPI_Comm comm, const char * filename, int amode, MPI_Info info, MPI_File * fh)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_preallocate:with:">MPI_File_preallocate: fh with: size	&lt;C: int __stdcall  MPI_File_preallocate(MPI_File fh, MPI_Offset size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read:with:with:with:with:">MPI_File_read: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_read(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_all:with:with:with:with:">MPI_File_read_all: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_read_all(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_all_begin:with:with:with:">MPI_File_read_all_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  MPI_File_read_all_begin(MPI_File fh, void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_all_end:with:with:">MPI_File_read_all_end: fh with: buf with: status	&lt;C: int __stdcall  MPI_File_read_all_end(MPI_File fh, void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_at:with:with:with:with:with:">MPI_File_read_at: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_read_at(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_at_all:with:with:with:with:with:">MPI_File_read_at_all: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_at_all_begin:with:with:with:with:">MPI_File_read_at_all_begin: fh with: offset with: buf with: count with: datatype	&lt;C: int __stdcall  MPI_File_read_at_all_begin(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_at_all_end:with:with:">MPI_File_read_at_all_end: fh with: buf with: status	&lt;C: int __stdcall  MPI_File_read_at_all_end(MPI_File fh, void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_ordered:with:with:with:with:">MPI_File_read_ordered: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_read_ordered(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_ordered_begin:with:with:with:">MPI_File_read_ordered_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  MPI_File_read_ordered_begin(MPI_File fh, void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_ordered_end:with:with:">MPI_File_read_ordered_end: fh with: buf with: status	&lt;C: int __stdcall  MPI_File_read_ordered_end(MPI_File fh, void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_read_shared:with:with:with:with:">MPI_File_read_shared: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_read_shared(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_seek:with:with:">MPI_File_seek: fh with: offset with: whence	&lt;C: int __stdcall  MPI_File_seek(MPI_File fh, MPI_Offset offset, int whence)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_seek_shared:with:with:">MPI_File_seek_shared: fh with: offset with: whence	&lt;C: int __stdcall  MPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_set_atomicity:with:">MPI_File_set_atomicity: fh with: flag	&lt;C: int __stdcall  MPI_File_set_atomicity(MPI_File fh, int flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_set_errhandler:with:">MPI_File_set_errhandler: file with: errhandler	&lt;C: int __stdcall  MPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_set_info:with:">MPI_File_set_info: fh with: info	&lt;C: int __stdcall  MPI_File_set_info(MPI_File fh, MPI_Info info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_set_size:with:">MPI_File_set_size: fh with: size	&lt;C: int __stdcall  MPI_File_set_size(MPI_File fh, MPI_Offset size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_set_view:with:with:with:with:with:">MPI_File_set_view: fh with: disp with: etype with: filetype with: datarep with: info	&lt;C: int __stdcall  MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char * datarep, MPI_Info info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_sync:">MPI_File_sync: fh	&lt;C: int __stdcall  MPI_File_sync(MPI_File fh)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write:with:with:with:with:">MPI_File_write: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_write(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_all:with:with:with:with:">MPI_File_write_all: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_write_all(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_all_begin:with:with:with:">MPI_File_write_all_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  MPI_File_write_all_begin(MPI_File fh, const void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_all_end:with:with:">MPI_File_write_all_end: fh with: buf with: status	&lt;C: int __stdcall  MPI_File_write_all_end(MPI_File fh, const void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_at:with:with:with:with:with:">MPI_File_write_at: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_at_all:with:with:with:with:with:">MPI_File_write_at_all: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_at_all_begin:with:with:with:with:">MPI_File_write_at_all_begin: fh with: offset with: buf with: count with: datatype	&lt;C: int __stdcall  MPI_File_write_at_all_begin(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_at_all_end:with:with:">MPI_File_write_at_all_end: fh with: buf with: status	&lt;C: int __stdcall  MPI_File_write_at_all_end(MPI_File fh, const void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_ordered:with:with:with:with:">MPI_File_write_ordered: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_write_ordered(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_ordered_begin:with:with:with:">MPI_File_write_ordered_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  MPI_File_write_ordered_begin(MPI_File fh, const void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_ordered_end:with:with:">MPI_File_write_ordered_end: fh with: buf with: status	&lt;C: int __stdcall  MPI_File_write_ordered_end(MPI_File fh, const void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_File_write_shared:with:with:with:with:">MPI_File_write_shared: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  MPI_File_write_shared(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Finalize">MPI_Finalize	&lt;C: int __stdcall  MPI_Finalize(void)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Finalized:">MPI_Finalized: flag	&lt;C: int __stdcall  MPI_Finalized(int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Free_mem:">MPI_Free_mem: base	&lt;C: int __stdcall  MPI_Free_mem(void * base)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Gather:with:with:with:with:with:with:with:">MPI_Gather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm	&lt;C: int __stdcall  MPI_Gather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Gatherv:with:with:with:with:with:with:with:with:">MPI_Gatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: root with: comm	&lt;C: int __stdcall  MPI_Gatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get:with:with:with:with:with:with:with:">MPI_Get: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win	&lt;C: int __stdcall  MPI_Get(void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_accumulate:with:with:with:with:with:with:with:with:with:with:with:">MPI_Get_accumulate: origin_addr with: origin_count with: origin_datatype with: result_addr with: result_count with: result_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win	&lt;C: int __stdcall  MPI_Get_accumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, void * result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_address:with:">MPI_Get_address: location with: address	&lt;C: int __stdcall  MPI_Get_address(const void * location, MPI_Aint * address)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_count:with:with:">MPI_Get_count: status with: datatype with: count	&lt;C: int __stdcall  MPI_Get_count(const MPI_Status * status, MPI_Datatype datatype, int * count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_elements:with:with:">MPI_Get_elements: status with: datatype with: count	&lt;C: int __stdcall  MPI_Get_elements(const MPI_Status * status, MPI_Datatype datatype, int * count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_elements_x:with:with:">MPI_Get_elements_x: status with: datatype with: count	&lt;C: int __stdcall  MPI_Get_elements_x(const MPI_Status * status, MPI_Datatype datatype, MPI_Count * count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_library_version:with:">MPI_Get_library_version: version with: resultlen	&lt;C: int __stdcall  MPI_Get_library_version(char * version, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_processor_name:with:">MPI_Get_processor_name: name with: resultlen	&lt;C: int __stdcall  MPI_Get_processor_name(char * name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Get_version:with:">MPI_Get_version: version with: subversion	&lt;C: int __stdcall  MPI_Get_version(int * version, int * subversion)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Graph_create:with:with:with:with:with:">MPI_Graph_create: comm_old with: nnodes with: index with: edges with: reorder with: comm_graph	&lt;C: int __stdcall  MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int index[], const int edges[], int reorder, MPI_Comm * comm_graph)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Graph_get:with:with:with:with:">MPI_Graph_get: comm with: maxindex with: maxedges with: index with: edges	&lt;C: int __stdcall  MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int index[], int edges[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Graph_map:with:with:with:with:">MPI_Graph_map: comm with: nnodes with: index with: edges with: newrank	&lt;C: int __stdcall  MPI_Graph_map(MPI_Comm comm, int nnodes, const int index[], const int edges[], int * newrank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Graph_neighbors:with:with:with:">MPI_Graph_neighbors: comm with: rank with: maxneighbors with: neighbors	&lt;C: int __stdcall  MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Graph_neighbors_count:with:with:">MPI_Graph_neighbors_count: comm with: rank with: nneighbors	&lt;C: int __stdcall  MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int * nneighbors)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Graphdims_get:with:with:">MPI_Graphdims_get: comm with: nnodes with: nedges	&lt;C: int __stdcall  MPI_Graphdims_get(MPI_Comm comm, int * nnodes, int * nedges)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Grequest_complete:">MPI_Grequest_complete: request	&lt;C: int __stdcall  MPI_Grequest_complete(MPI_Request request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Grequest_start:with:with:with:with:">MPI_Grequest_start: query_fn with: free_fn with: cancel_fn with: extra_state with: request	&lt;C: int __stdcall  MPI_Grequest_start(int ( __stdcall * query_fn)(void * extra_state, MPI_Status * status), int ( __stdcall * free_fn)(void * extra_state), int ( __stdcall * cancel_fn)(void * extra_state, int complete), void * extra_state, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_compare:with:with:">MPI_Group_compare: group1 with: group2 with: result	&lt;C: int __stdcall  MPI_Group_compare(MPI_Group group1, MPI_Group group2, int * result)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_difference:with:with:">MPI_Group_difference: group1 with: group2 with: newgroup	&lt;C: int __stdcall  MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_excl:with:with:with:">MPI_Group_excl: group with: n with: ranks with: newgroup	&lt;C: int __stdcall  MPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_free:">MPI_Group_free: group	&lt;C: int __stdcall  MPI_Group_free(MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_incl:with:with:with:">MPI_Group_incl: group with: n with: ranks with: newgroup	&lt;C: int __stdcall  MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_intersection:with:with:">MPI_Group_intersection: group1 with: group2 with: newgroup	&lt;C: int __stdcall  MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_range_excl:with:with:with:">MPI_Group_range_excl: group with: n with: ranges with: newgroup	&lt;C: int __stdcall  MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_range_incl:with:with:with:">MPI_Group_range_incl: group with: n with: ranges with: newgroup	&lt;C: int __stdcall  MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_rank:with:">MPI_Group_rank: group with: rank	&lt;C: int __stdcall  MPI_Group_rank(MPI_Group group, int * rank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_size:with:">MPI_Group_size: group with: size	&lt;C: int __stdcall  MPI_Group_size(MPI_Group group, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_translate_ranks:with:with:with:with:">MPI_Group_translate_ranks: group1 with: n with: ranks1 with: group2 with: ranks2	&lt;C: int __stdcall  MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Group_union:with:with:">MPI_Group_union: group1 with: group2 with: newgroup	&lt;C: int __stdcall  MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iallgather:with:with:with:with:with:with:with:">MPI_Iallgather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm with: request	&lt;C: int __stdcall  MPI_Iallgather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iallgatherv:with:with:with:with:with:with:with:with:">MPI_Iallgatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: comm with: request	&lt;C: int __stdcall  MPI_Iallgatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iallreduce:with:with:with:with:with:with:">MPI_Iallreduce: sendbuf with: recvbuf with: count with: datatype with: op with: comm with: request	&lt;C: int __stdcall  MPI_Iallreduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ialltoall:with:with:with:with:with:with:with:">MPI_Ialltoall: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm with: request	&lt;C: int __stdcall  MPI_Ialltoall(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ialltoallv:with:with:with:with:with:with:with:with:with:">MPI_Ialltoallv: sendbuf with: sendcounts with: sdispls with: sendtype with: recvbuf with: recvcounts with: rdispls with: recvtype with: comm with: request	&lt;C: int __stdcall  MPI_Ialltoallv(const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ialltoallw:with:with:with:with:with:with:with:with:with:">MPI_Ialltoallw: sendbuf with: sendcounts with: sdispls with: sendtypes with: recvbuf with: recvcounts with: rdispls with: recvtypes with: comm with: request	&lt;C: int __stdcall  MPI_Ialltoallw(const void * sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ibarrier:with:">MPI_Ibarrier: comm with: request	&lt;C: int __stdcall  MPI_Ibarrier(MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ibcast:with:with:with:with:with:">MPI_Ibcast: buffer with: count with: datatype with: root with: comm with: request	&lt;C: int __stdcall  MPI_Ibcast(void * buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ibsend:with:with:with:with:with:with:">MPI_Ibsend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Ibsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iexscan:with:with:with:with:with:with:">MPI_Iexscan: sendbuf with: recvbuf with: count with: datatype with: op with: comm with: request	&lt;C: int __stdcall  MPI_Iexscan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Igather:with:with:with:with:with:with:with:with:">MPI_Igather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  MPI_Igather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Igatherv:with:with:with:with:with:with:with:with:with:">MPI_Igatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  MPI_Igatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Improbe:with:with:with:with:with:">MPI_Improbe: source with: tag with: comm with: flag with: message with: status	&lt;C: int __stdcall  MPI_Improbe(int source, int tag, MPI_Comm comm, int * flag, MPI_Message * message, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Imrecv:with:with:with:with:">MPI_Imrecv: buf with: count with: datatype with: message with: request	&lt;C: int __stdcall  MPI_Imrecv(void * buf, int count, MPI_Datatype datatype, MPI_Message * message, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_create:">MPI_Info_create: info	&lt;C: int __stdcall  MPI_Info_create(MPI_Info * info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_delete:with:">MPI_Info_delete: info with: key	&lt;C: int __stdcall  MPI_Info_delete(MPI_Info info, const char * key)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_dup:with:">MPI_Info_dup: info with: newinfo	&lt;C: int __stdcall  MPI_Info_dup(MPI_Info info, MPI_Info * newinfo)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_free:">MPI_Info_free: info	&lt;C: int __stdcall  MPI_Info_free(MPI_Info * info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_get:with:with:with:with:">MPI_Info_get: info with: key with: valuelen with: value with: flag	&lt;C: int __stdcall  MPI_Info_get(MPI_Info info, const char * key, int valuelen, char * value, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_get_nkeys:with:">MPI_Info_get_nkeys: info with: nkeys	&lt;C: int __stdcall  MPI_Info_get_nkeys(MPI_Info info, int * nkeys)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_get_nthkey:with:with:">MPI_Info_get_nthkey: info with: n with: key	&lt;C: int __stdcall  MPI_Info_get_nthkey(MPI_Info info, int n, char * key)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_get_valuelen:with:with:with:">MPI_Info_get_valuelen: info with: key with: valuelen with: flag	&lt;C: int __stdcall  MPI_Info_get_valuelen(MPI_Info info, const char * key, int * valuelen, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Info_set:with:with:">MPI_Info_set: info with: key with: value	&lt;C: int __stdcall  MPI_Info_set(MPI_Info info, const char * key, const char * value)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Init:with:">MPI_Init: argc with: argv	&lt;C: int __stdcall  MPI_Init(const int * argc, char * * * argv)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Init_thread:with:with:with:">MPI_Init_thread: argc with: argv with: required with: provided	&lt;C: int __stdcall  MPI_Init_thread(const int * argc, char * * * argv, int required, int * provided)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Initialized:">MPI_Initialized: flag	&lt;C: int __stdcall  MPI_Initialized(int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Intercomm_create:with:with:with:with:with:">MPI_Intercomm_create: local_comm with: local_leader with: peer_comm with: remote_leader with: tag with: newintercomm	&lt;C: int __stdcall  MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm * newintercomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Intercomm_merge:with:with:">MPI_Intercomm_merge: intercomm with: high with: newintracomm	&lt;C: int __stdcall  MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm * newintracomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iprobe:with:with:with:with:">MPI_Iprobe: source with: tag with: comm with: flag with: status	&lt;C: int __stdcall  MPI_Iprobe(int source, int tag, MPI_Comm comm, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Irecv:with:with:with:with:with:with:">MPI_Irecv: buf with: count with: datatype with: source with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ireduce:with:with:with:with:with:with:with:">MPI_Ireduce: sendbuf with: recvbuf with: count with: datatype with: op with: root with: comm with: request	&lt;C: int __stdcall  MPI_Ireduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ireduce_scatter:with:with:with:with:with:with:">MPI_Ireduce_scatter: sendbuf with: recvbuf with: recvcounts with: datatype with: op with: comm with: request	&lt;C: int __stdcall  MPI_Ireduce_scatter(const void * sendbuf, void * recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ireduce_scatter_block:with:with:with:with:with:with:">MPI_Ireduce_scatter_block: sendbuf with: recvbuf with: recvcount with: datatype with: op with: comm with: request	&lt;C: int __stdcall  MPI_Ireduce_scatter_block(const void * sendbuf, void * recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Irsend:with:with:with:with:with:with:">MPI_Irsend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Irsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Is_thread_main:">MPI_Is_thread_main: flag	&lt;C: int __stdcall  MPI_Is_thread_main(int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iscan:with:with:with:with:with:with:">MPI_Iscan: sendbuf with: recvbuf with: count with: datatype with: op with: comm with: request	&lt;C: int __stdcall  MPI_Iscan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iscatter:with:with:with:with:with:with:with:with:">MPI_Iscatter: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  MPI_Iscatter(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Iscatterv:with:with:with:with:with:with:with:with:with:">MPI_Iscatterv: sendbuf with: sendcounts with: displs with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  MPI_Iscatterv(const void * sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Isend:with:with:with:with:with:with:">MPI_Isend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Isend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Issend:with:with:with:with:with:with:">MPI_Issend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Issend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Lookup_name:with:with:">MPI_Lookup_name: service_name with: info with: port_name	&lt;C: int __stdcall  MPI_Lookup_name(const char * service_name, MPI_Info info, char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Mprobe:with:with:with:with:">MPI_Mprobe: source with: tag with: comm with: message with: status	&lt;C: int __stdcall  MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message * message, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Mrecv:with:with:with:with:">MPI_Mrecv: buf with: count with: datatype with: message with: status	&lt;C: int __stdcall  MPI_Mrecv(void * buf, int count, MPI_Datatype datatype, MPI_Message * message, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Op_commutative:with:">MPI_Op_commutative: op with: commute	&lt;C: int __stdcall  MPI_Op_commutative(MPI_Op op, int * commute)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Op_create:with:with:">MPI_Op_create: user_fn with: commute with: op	&lt;C: int __stdcall  MPI_Op_create(void ( __stdcall * user_fn)(void * invec, void * inoutvec, int * len, MPI_Datatype * datatype), int commute, MPI_Op * op)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Op_free:">MPI_Op_free: op	&lt;C: int __stdcall  MPI_Op_free(MPI_Op * op)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Open_port:with:">MPI_Open_port: info with: port_name	&lt;C: int __stdcall  MPI_Open_port(MPI_Info info, char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Pack:with:with:with:with:with:with:">MPI_Pack: inbuf with: incount with: datatype with: outbuf with: outsize with: position with: comm	&lt;C: int __stdcall  MPI_Pack(const void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, int outsize, int * position, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Pack_external:with:with:with:with:with:with:">MPI_Pack_external: datarep with: inbuf with: incount with: datatype with: outbuf with: outsize with: position	&lt;C: int __stdcall  MPI_Pack_external(const char * datarep, const void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, MPI_Aint outsize, MPI_Aint * position)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Pack_external_size:with:with:with:">MPI_Pack_external_size: datarep with: incount with: datatype with: size	&lt;C: int __stdcall  MPI_Pack_external_size(const char * datarep, int incount, MPI_Datatype datatype, MPI_Aint * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Pack_size:with:with:with:">MPI_Pack_size: incount with: datatype with: comm with: size	&lt;C: int __stdcall  MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Probe:with:with:with:">MPI_Probe: source with: tag with: comm with: status	&lt;C: int __stdcall  MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Publish_name:with:with:">MPI_Publish_name: service_name with: info with: port_name	&lt;C: int __stdcall  MPI_Publish_name(const char * service_name, MPI_Info info, const char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Put:with:with:with:with:with:with:with:">MPI_Put: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win	&lt;C: int __stdcall  MPI_Put(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Query_thread:">MPI_Query_thread: provided	&lt;C: int __stdcall  MPI_Query_thread(int * provided)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Raccumulate:with:with:with:with:with:with:with:with:with:">MPI_Raccumulate: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win with: request	&lt;C: int __stdcall  MPI_Raccumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Recv:with:with:with:with:with:with:">MPI_Recv: buf with: count with: datatype with: source with: tag with: comm with: status	&lt;C: int __stdcall  MPI_Recv(void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Recv_init:with:with:with:with:with:with:">MPI_Recv_init: buf with: count with: datatype with: source with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Recv_init(void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Reduce:with:with:with:with:with:with:">MPI_Reduce: sendbuf with: recvbuf with: count with: datatype with: op with: root with: comm	&lt;C: int __stdcall  MPI_Reduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Reduce_local:with:with:with:with:">MPI_Reduce_local: inbuf with: inoutbuf with: count with: datatype with: op	&lt;C: int __stdcall  MPI_Reduce_local(const void * inbuf, void * inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Reduce_scatter:with:with:with:with:with:">MPI_Reduce_scatter: sendbuf with: recvbuf with: recvcounts with: datatype with: op with: comm	&lt;C: int __stdcall  MPI_Reduce_scatter(const void * sendbuf, void * recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Reduce_scatter_block:with:with:with:with:with:">MPI_Reduce_scatter_block: sendbuf with: recvbuf with: recvcount with: datatype with: op with: comm	&lt;C: int __stdcall  MPI_Reduce_scatter_block(const void * sendbuf, void * recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Register_datarep:with:with:with:with:">MPI_Register_datarep: datarep with: read_conversion_fn with: write_conversion_fn with: dtype_file_extent_fn with: extra_state	&lt;C: int __stdcall  MPI_Register_datarep(const char * datarep, int ( __stdcall * read_conversion_fn)(void * userbuf, MPI_Datatype datatype, int count, void * filebuf, MPI_Offset position, void * extra_state), int ( __stdcall * write_conversion_fn)(void * userbuf, MPI_Datatype datatype, int count, void * filebuf, MPI_Offset position, void * extra_state), int ( __stdcall * dtype_file_extent_fn)(MPI_Datatype datatype, MPI_Aint * file_extent, void * extra_state), void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Request_free:">MPI_Request_free: request	&lt;C: int __stdcall  MPI_Request_free(MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Request_get_status:with:with:">MPI_Request_get_status: request with: flag with: status	&lt;C: int __stdcall  MPI_Request_get_status(MPI_Request request, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Rget:with:with:with:with:with:with:with:with:">MPI_Rget: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win with: request	&lt;C: int __stdcall  MPI_Rget(void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Rget_accumulate:with:with:with:with:with:with:with:with:with:with:with:with:">MPI_Rget_accumulate: origin_addr with: origin_count with: origin_datatype with: result_addr with: result_count with: result_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win with: request	&lt;C: int __stdcall  MPI_Rget_accumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, void * result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Rput:with:with:with:with:with:with:with:with:">MPI_Rput: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win with: request	&lt;C: int __stdcall  MPI_Rput(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Rsend:with:with:with:with:with:">MPI_Rsend: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  MPI_Rsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Rsend_init:with:with:with:with:with:with:">MPI_Rsend_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Rsend_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Scan:with:with:with:with:with:">MPI_Scan: sendbuf with: recvbuf with: count with: datatype with: op with: comm	&lt;C: int __stdcall  MPI_Scan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Scatter:with:with:with:with:with:with:with:">MPI_Scatter: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm	&lt;C: int __stdcall  MPI_Scatter(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Scatterv:with:with:with:with:with:with:with:with:">MPI_Scatterv: sendbuf with: sendcounts with: displs with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm	&lt;C: int __stdcall  MPI_Scatterv(const void * sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Send:with:with:with:with:with:">MPI_Send: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  MPI_Send(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Send_init:with:with:with:with:with:with:">MPI_Send_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Send_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Sendrecv:with:with:with:with:with:with:with:with:with:with:with:">MPI_Sendrecv: sendbuf with: sendcount with: sendtype with: dest with: sendtag with: recvbuf with: recvcount with: recvtype with: source with: recvtag with: comm with: status	&lt;C: int __stdcall  MPI_Sendrecv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void * recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Sendrecv_replace:with:with:with:with:with:with:with:with:">MPI_Sendrecv_replace: buf with: count with: datatype with: dest with: sendtag with: source with: recvtag with: comm with: status	&lt;C: int __stdcall  MPI_Sendrecv_replace(void * buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ssend:with:with:with:with:with:">MPI_Ssend: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  MPI_Ssend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Ssend_init:with:with:with:with:with:with:">MPI_Ssend_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  MPI_Ssend_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Start:">MPI_Start: request	&lt;C: int __stdcall  MPI_Start(MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Startall:with:">MPI_Startall: count with: array_of_requests	&lt;C: int __stdcall  MPI_Startall(int count, MPI_Request array_of_requests[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Status_set_cancelled:with:">MPI_Status_set_cancelled: status with: flag	&lt;C: int __stdcall  MPI_Status_set_cancelled(MPI_Status * status, int flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Status_set_elements:with:with:">MPI_Status_set_elements: status with: datatype with: count	&lt;C: int __stdcall  MPI_Status_set_elements(MPI_Status * status, MPI_Datatype datatype, int count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Status_set_elements_x:with:with:">MPI_Status_set_elements_x: status with: datatype with: count	&lt;C: int __stdcall  MPI_Status_set_elements_x(MPI_Status * status, MPI_Datatype datatype, MPI_Count count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Test:with:with:">MPI_Test: request with: flag with: status	&lt;C: int __stdcall  MPI_Test(MPI_Request * request, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Test_cancelled:with:">MPI_Test_cancelled: status with: flag	&lt;C: int __stdcall  MPI_Test_cancelled(const MPI_Status * status, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Testall:with:with:with:">MPI_Testall: count with: array_of_requests with: flag with: array_of_statuses	&lt;C: int __stdcall  MPI_Testall(int count, MPI_Request array_of_requests[], int * flag, MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Testany:with:with:with:with:">MPI_Testany: count with: array_of_requests with: index with: flag with: status	&lt;C: int __stdcall  MPI_Testany(int count, MPI_Request array_of_requests[], int * index, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Testsome:with:with:with:with:">MPI_Testsome: incount with: array_of_requests with: outcount with: array_of_indices with: array_of_statuses	&lt;C: int __stdcall  MPI_Testsome(int incount, MPI_Request array_of_requests[], int * outcount, int array_of_indices[], MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Topo_test:with:">MPI_Topo_test: comm with: status	&lt;C: int __stdcall  MPI_Topo_test(MPI_Comm comm, int * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_commit:">MPI_Type_commit: datatype	&lt;C: int __stdcall  MPI_Type_commit(MPI_Datatype * datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_contiguous:with:with:">MPI_Type_contiguous: count with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_darray:with:with:with:with:with:with:with:with:with:">MPI_Type_create_darray: size with: rank with: ndims with: array_of_gsizes with: array_of_distribs with: array_of_dargs with: array_of_psizes with: order with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[], const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_hindexed:with:with:with:with:">MPI_Type_create_hindexed: count with: array_of_blocklengths with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_create_hindexed(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_hindexed_block:with:with:with:with:">MPI_Type_create_hindexed_block: count with: blocklength with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_create_hindexed_block(int count, int blocklength, const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_hvector:with:with:with:with:">MPI_Type_create_hvector: count with: blocklength with: stride with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_indexed_block:with:with:with:with:">MPI_Type_create_indexed_block: count with: blocklength with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_keyval:with:with:with:">MPI_Type_create_keyval: type_copy_attr_fn with: type_delete_attr_fn with: type_keyval with: extra_state	&lt;C: int __stdcall  MPI_Type_create_keyval(int ( __stdcall * type_copy_attr_fn)(MPI_Datatype olddatatype, int datatype_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag), int ( __stdcall * type_delete_attr_fn)(MPI_Datatype datatype, int datatype_keyval, void * attribute_val, void * extra_state), int * type_keyval, void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_resized:with:with:with:">MPI_Type_create_resized: oldtype with: lb with: extent with: newtype	&lt;C: int __stdcall  MPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_struct:with:with:with:with:">MPI_Type_create_struct: count with: array_of_blocklengths with: array_of_displacements with: array_of_types with: newtype	&lt;C: int __stdcall  MPI_Type_create_struct(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_create_subarray:with:with:with:with:with:with:">MPI_Type_create_subarray: ndims with: array_of_sizes with: array_of_subsizes with: array_of_starts with: order with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_delete_attr:with:">MPI_Type_delete_attr: type with: type_keyval	&lt;C: int __stdcall  MPI_Type_delete_attr(MPI_Datatype type, int type_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_dup:with:">MPI_Type_dup: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_free:">MPI_Type_free: datatype	&lt;C: int __stdcall  MPI_Type_free(MPI_Datatype * datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_free_keyval:">MPI_Type_free_keyval: type_keyval	&lt;C: int __stdcall  MPI_Type_free_keyval(int * type_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_attr:with:with:with:">MPI_Type_get_attr: type with: type_keyval with: attribute_val with: flag	&lt;C: int __stdcall  MPI_Type_get_attr(MPI_Datatype type, int type_keyval, void * attribute_val, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_contents:with:with:with:with:with:with:">MPI_Type_get_contents: datatype with: max_integers with: max_addresses with: max_datatypes with: array_of_integers with: array_of_addresses with: array_of_datatypes	&lt;C: int __stdcall  MPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_envelope:with:with:with:with:">MPI_Type_get_envelope: datatype with: num_integers with: num_addresses with: num_datatypes with: combiner	&lt;C: int __stdcall  MPI_Type_get_envelope(MPI_Datatype datatype, int * num_integers, int * num_addresses, int * num_datatypes, int * combiner)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_extent:with:with:">MPI_Type_get_extent: datatype with: lb with: extent	&lt;C: int __stdcall  MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint * lb, MPI_Aint * extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_extent_x:with:with:">MPI_Type_get_extent_x: datatype with: lb with: extent	&lt;C: int __stdcall  MPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count * lb, MPI_Count * extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_name:with:with:">MPI_Type_get_name: datatype with: type_name with: resultlen	&lt;C: int __stdcall  MPI_Type_get_name(MPI_Datatype datatype, char * type_name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_true_extent:with:with:">MPI_Type_get_true_extent: datatype with: true_lb with: true_extent	&lt;C: int __stdcall  MPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint * true_lb, MPI_Aint * true_extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_get_true_extent_x:with:with:">MPI_Type_get_true_extent_x: datatype with: true_lb with: true_extent	&lt;C: int __stdcall  MPI_Type_get_true_extent_x(MPI_Datatype datatype, MPI_Count * true_lb, MPI_Count * true_extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_indexed:with:with:with:with:">MPI_Type_indexed: count with: array_of_blocklengths with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_indexed(int count, const int array_of_blocklengths[], const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_set_attr:with:with:">MPI_Type_set_attr: type with: type_keyval with: attribute_val	&lt;C: int __stdcall  MPI_Type_set_attr(MPI_Datatype type, int type_keyval, void * attribute_val)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_set_name:with:">MPI_Type_set_name: datatype with: type_name	&lt;C: int __stdcall  MPI_Type_set_name(MPI_Datatype datatype, const char * type_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_size:with:">MPI_Type_size: datatype with: size	&lt;C: int __stdcall  MPI_Type_size(MPI_Datatype datatype, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_size_x:with:">MPI_Type_size_x: datatype with: size	&lt;C: int __stdcall  MPI_Type_size_x(MPI_Datatype datatype, MPI_Count * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Type_vector:with:with:with:with:">MPI_Type_vector: count with: blocklength with: stride with: oldtype with: newtype	&lt;C: int __stdcall  MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Unpack:with:with:with:with:with:with:">MPI_Unpack: inbuf with: insize with: position with: outbuf with: outcount with: datatype with: comm	&lt;C: int __stdcall  MPI_Unpack(const void * inbuf, int insize, int * position, void * outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Unpack_external:with:with:with:with:with:with:">MPI_Unpack_external: datarep with: inbuf with: insize with: position with: outbuf with: outcount with: datatype	&lt;C: int __stdcall  MPI_Unpack_external(const char * datarep, const void * inbuf, MPI_Aint insize, MPI_Aint * position, void * outbuf, int outcount, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Unpublish_name:with:with:">MPI_Unpublish_name: service_name with: info with: port_name	&lt;C: int __stdcall  MPI_Unpublish_name(const char * service_name, MPI_Info info, const char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Wait:with:">MPI_Wait: request with: status	&lt;C: int __stdcall  MPI_Wait(MPI_Request * request, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Waitall:with:with:">MPI_Waitall: count with: array_of_requests with: array_of_statuses	&lt;C: int __stdcall  MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Waitany:with:with:with:">MPI_Waitany: count with: array_of_requests with: index with: status	&lt;C: int __stdcall  MPI_Waitany(int count, MPI_Request array_of_requests[], int * index, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Waitsome:with:with:with:with:">MPI_Waitsome: incount with: array_of_requests with: outcount with: array_of_indices with: array_of_statuses	&lt;C: int __stdcall  MPI_Waitsome(int incount, MPI_Request array_of_requests[], int * outcount, int array_of_indices[], MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_allocate:with:with:with:with:with:">MPI_Win_allocate: size with: disp_unit with: info with: comm with: baseptr with: win	&lt;C: int __stdcall  MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void * baseptr, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_allocate_shared:with:with:with:with:with:">MPI_Win_allocate_shared: size with: disp_unit with: info with: comm with: baseptr with: win	&lt;C: int __stdcall  MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void * baseptr, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_attach:with:with:">MPI_Win_attach: win with: base with: size	&lt;C: int __stdcall  MPI_Win_attach(MPI_Win win, void * base, MPI_Aint size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_call_errhandler:with:">MPI_Win_call_errhandler: win with: errorcode	&lt;C: int __stdcall  MPI_Win_call_errhandler(MPI_Win win, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_complete:">MPI_Win_complete: win	&lt;C: int __stdcall  MPI_Win_complete(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_create:with:with:with:with:with:">MPI_Win_create: base with: size with: disp_unit with: info with: comm with: win	&lt;C: int __stdcall  MPI_Win_create(void * base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_create_dynamic:with:with:">MPI_Win_create_dynamic: info with: comm with: win	&lt;C: int __stdcall  MPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_create_errhandler:with:">MPI_Win_create_errhandler: function with: errhandler	&lt;C: int __stdcall  MPI_Win_create_errhandler(void ( __stdcall * function)(MPI_Win * win, int * errcode, ...), MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_create_keyval:with:with:with:">MPI_Win_create_keyval: win_copy_attr_fn with: win_delete_attr_fn with: win_keyval with: extra_state	&lt;C: int __stdcall  MPI_Win_create_keyval(int ( __stdcall * win_copy_attr_fn)(MPI_Win oldwin, int win_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag), int ( __stdcall * win_delete_attr_fn)(MPI_Win win, int win_keyval, void * attribute_val, void * extra_state), int * win_keyval, void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_delete_attr:with:">MPI_Win_delete_attr: win with: win_keyval	&lt;C: int __stdcall  MPI_Win_delete_attr(MPI_Win win, int win_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_detach:with:">MPI_Win_detach: win with: base	&lt;C: int __stdcall  MPI_Win_detach(MPI_Win win, void * base)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_fence:with:">MPI_Win_fence: assert with: win	&lt;C: int __stdcall  MPI_Win_fence(int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_flush:with:">MPI_Win_flush: rank with: win	&lt;C: int __stdcall  MPI_Win_flush(int rank, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_flush_all:">MPI_Win_flush_all: win	&lt;C: int __stdcall  MPI_Win_flush_all(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_flush_local:with:">MPI_Win_flush_local: rank with: win	&lt;C: int __stdcall  MPI_Win_flush_local(int rank, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_flush_local_all:">MPI_Win_flush_local_all: win	&lt;C: int __stdcall  MPI_Win_flush_local_all(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_free:">MPI_Win_free: win	&lt;C: int __stdcall  MPI_Win_free(MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_free_keyval:">MPI_Win_free_keyval: win_keyval	&lt;C: int __stdcall  MPI_Win_free_keyval(int * win_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_get_attr:with:with:with:">MPI_Win_get_attr: win with: win_keyval with: attribute_val with: flag	&lt;C: int __stdcall  MPI_Win_get_attr(MPI_Win win, int win_keyval, void * attribute_val, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_get_errhandler:with:">MPI_Win_get_errhandler: win with: errhandler	&lt;C: int __stdcall  MPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_get_group:with:">MPI_Win_get_group: win with: group	&lt;C: int __stdcall  MPI_Win_get_group(MPI_Win win, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_get_name:with:with:">MPI_Win_get_name: win with: win_name with: resultlen	&lt;C: int __stdcall  MPI_Win_get_name(MPI_Win win, char * win_name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_lock:with:with:with:">MPI_Win_lock: lock_type with: rank with: assert with: win	&lt;C: int __stdcall  MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_lock_all:with:">MPI_Win_lock_all: assert with: win	&lt;C: int __stdcall  MPI_Win_lock_all(int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_post:with:with:">MPI_Win_post: group with: assert with: win	&lt;C: int __stdcall  MPI_Win_post(MPI_Group group, int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_set_attr:with:with:">MPI_Win_set_attr: win with: win_keyval with: attribute_val	&lt;C: int __stdcall  MPI_Win_set_attr(MPI_Win win, int win_keyval, void * attribute_val)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_set_errhandler:with:">MPI_Win_set_errhandler: win with: errhandler	&lt;C: int __stdcall  MPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_set_name:with:">MPI_Win_set_name: win with: win_name	&lt;C: int __stdcall  MPI_Win_set_name(MPI_Win win, const char * win_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_shared_query:with:with:with:with:">MPI_Win_shared_query: win with: rank with: size with: disp_unit with: baseptr	&lt;C: int __stdcall  MPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint * size, int * disp_unit, void * baseptr)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_start:with:with:">MPI_Win_start: group with: assert with: win	&lt;C: int __stdcall  MPI_Win_start(MPI_Group group, int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_sync:">MPI_Win_sync: win	&lt;C: int __stdcall  MPI_Win_sync(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_test:with:">MPI_Win_test: win with: flag	&lt;C: int __stdcall  MPI_Win_test(MPI_Win win, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_unlock:with:">MPI_Win_unlock: rank with: win	&lt;C: int __stdcall  MPI_Win_unlock(int rank, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_unlock_all:">MPI_Win_unlock_all: win	&lt;C: int __stdcall  MPI_Win_unlock_all(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Win_wait:">MPI_Win_wait: win	&lt;C: int __stdcall  MPI_Win_wait(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Wtick">MPI_Wtick	&lt;C: double __stdcall  MPI_Wtick(void)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="MPI_Wtime">MPI_Wtime	&lt;C: double __stdcall  MPI_Wtime(void)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Abort:with:">PMPI_Abort: comm with: errorcode	&lt;C: int __stdcall  PMPI_Abort(MPI_Comm comm, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Accumulate:with:with:with:with:with:with:with:with:">PMPI_Accumulate: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win	&lt;C: int __stdcall  PMPI_Accumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Add_error_class:">PMPI_Add_error_class: errorclass	&lt;C: int __stdcall  PMPI_Add_error_class(int * errorclass)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Add_error_code:with:">PMPI_Add_error_code: errorclass with: errorcode	&lt;C: int __stdcall  PMPI_Add_error_code(int errorclass, int * errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Add_error_string:with:">PMPI_Add_error_string: errorcode with: string	&lt;C: int __stdcall  PMPI_Add_error_string(int errorcode, const char * string)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Allgather:with:with:with:with:with:with:">PMPI_Allgather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm	&lt;C: int __stdcall  PMPI_Allgather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Allgatherv:with:with:with:with:with:with:with:">PMPI_Allgatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: comm	&lt;C: int __stdcall  PMPI_Allgatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Alloc_mem:with:with:">PMPI_Alloc_mem: size with: info with: baseptr	&lt;C: int __stdcall  PMPI_Alloc_mem(MPI_Aint size, MPI_Info info, void * baseptr)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Allreduce:with:with:with:with:with:">PMPI_Allreduce: sendbuf with: recvbuf with: count with: datatype with: op with: comm	&lt;C: int __stdcall  PMPI_Allreduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Alltoall:with:with:with:with:with:with:">PMPI_Alltoall: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm	&lt;C: int __stdcall  PMPI_Alltoall(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Alltoallv:with:with:with:with:with:with:with:with:">PMPI_Alltoallv: sendbuf with: sendcounts with: sdispls with: sendtype with: recvbuf with: recvcounts with: rdispls with: recvtype with: comm	&lt;C: int __stdcall  PMPI_Alltoallv(const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Alltoallw:with:with:with:with:with:with:with:with:">PMPI_Alltoallw: sendbuf with: sendcounts with: sdispls with: sendtypes with: recvbuf with: recvcounts with: rdispls with: recvtypes with: comm	&lt;C: int __stdcall  PMPI_Alltoallw(const void * sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Barrier:">PMPI_Barrier: comm	&lt;C: int __stdcall  PMPI_Barrier(MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Bcast:with:with:with:with:">PMPI_Bcast: buffer with: count with: datatype with: root with: comm	&lt;C: int __stdcall  PMPI_Bcast(void * buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Bsend:with:with:with:with:with:">PMPI_Bsend: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  PMPI_Bsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Bsend_init:with:with:with:with:with:with:">PMPI_Bsend_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Bsend_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Buffer_attach:with:">PMPI_Buffer_attach: buffer with: size	&lt;C: int __stdcall  PMPI_Buffer_attach(void * buffer, int size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Buffer_detach:with:">PMPI_Buffer_detach: buffer_addr with: size	&lt;C: int __stdcall  PMPI_Buffer_detach(void * buffer_addr, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cancel:">PMPI_Cancel: request	&lt;C: int __stdcall  PMPI_Cancel(MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_coords:with:with:with:">PMPI_Cart_coords: comm with: rank with: maxdims with: coords	&lt;C: int __stdcall  PMPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_create:with:with:with:with:with:">PMPI_Cart_create: comm_old with: ndims with: dims with: periods with: reorder with: comm_cart	&lt;C: int __stdcall  PMPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm * comm_cart)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_get:with:with:with:with:">PMPI_Cart_get: comm with: maxdims with: dims with: periods with: coords	&lt;C: int __stdcall  PMPI_Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_map:with:with:with:with:">PMPI_Cart_map: comm with: ndims with: dims with: periods with: newrank	&lt;C: int __stdcall  PMPI_Cart_map(MPI_Comm comm, int ndims, const int dims[], const int periods[], int * newrank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_rank:with:with:">PMPI_Cart_rank: comm with: coords with: rank	&lt;C: int __stdcall  PMPI_Cart_rank(MPI_Comm comm, const int coords[], int * rank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_shift:with:with:with:with:">PMPI_Cart_shift: comm with: direction with: disp with: rank_source with: rank_dest	&lt;C: int __stdcall  PMPI_Cart_shift(MPI_Comm comm, int direction, int disp, int * rank_source, int * rank_dest)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cart_sub:with:with:">PMPI_Cart_sub: comm with: remain_dims with: newcomm	&lt;C: int __stdcall  PMPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Cartdim_get:with:">PMPI_Cartdim_get: comm with: ndims	&lt;C: int __stdcall  PMPI_Cartdim_get(MPI_Comm comm, int * ndims)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Close_port:">PMPI_Close_port: port_name	&lt;C: int __stdcall  PMPI_Close_port(const char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_accept:with:with:with:with:">PMPI_Comm_accept: port_name with: info with: root with: comm with: newcomm	&lt;C: int __stdcall  PMPI_Comm_accept(const char * port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_call_errhandler:with:">PMPI_Comm_call_errhandler: comm with: errorcode	&lt;C: int __stdcall  PMPI_Comm_call_errhandler(MPI_Comm comm, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_compare:with:with:">PMPI_Comm_compare: comm1 with: comm2 with: result	&lt;C: int __stdcall  PMPI_Comm_compare(MPI_Comm comm1, MPI_Comm comm2, int * result)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_connect:with:with:with:with:">PMPI_Comm_connect: port_name with: info with: root with: comm with: newcomm	&lt;C: int __stdcall  PMPI_Comm_connect(const char * port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_create:with:with:">PMPI_Comm_create: comm with: group with: newcomm	&lt;C: int __stdcall  PMPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_create_errhandler:with:">PMPI_Comm_create_errhandler: function with: errhandler	&lt;C: int __stdcall  PMPI_Comm_create_errhandler(void ( __stdcall * function)(MPI_Comm * comm, int * errcode, ...), MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_create_keyval:with:with:with:">PMPI_Comm_create_keyval: comm_copy_attr_fn with: comm_delete_attr_fn with: comm_keyval with: extra_state	&lt;C: int __stdcall  PMPI_Comm_create_keyval(int ( __stdcall * comm_copy_attr_fn)(MPI_Comm oldcomm, int comm_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag), int ( __stdcall * comm_delete_attr_fn)(MPI_Comm comm, int comm_keyval, void * attribute_val, void * extra_state), int * comm_keyval, void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_delete_attr:with:">PMPI_Comm_delete_attr: comm with: comm_keyval	&lt;C: int __stdcall  PMPI_Comm_delete_attr(MPI_Comm comm, int comm_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_disconnect:">PMPI_Comm_disconnect: comm	&lt;C: int __stdcall  PMPI_Comm_disconnect(MPI_Comm * comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_dup:with:">PMPI_Comm_dup: comm with: newcomm	&lt;C: int __stdcall  PMPI_Comm_dup(MPI_Comm comm, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_free:">PMPI_Comm_free: comm	&lt;C: int __stdcall  PMPI_Comm_free(MPI_Comm * comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_free_keyval:">PMPI_Comm_free_keyval: comm_keyval	&lt;C: int __stdcall  PMPI_Comm_free_keyval(int * comm_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_get_attr:with:with:with:">PMPI_Comm_get_attr: comm with: comm_keyval with: attribute_val with: flag	&lt;C: int __stdcall  PMPI_Comm_get_attr(MPI_Comm comm, int comm_keyval, void * attribute_val, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_get_errhandler:with:">PMPI_Comm_get_errhandler: comm with: errhandler	&lt;C: int __stdcall  PMPI_Comm_get_errhandler(MPI_Comm comm, MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_get_name:with:with:">PMPI_Comm_get_name: comm with: comm_name with: resultlen	&lt;C: int __stdcall  PMPI_Comm_get_name(MPI_Comm comm, char * comm_name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_get_parent:">PMPI_Comm_get_parent: parent	&lt;C: int __stdcall  PMPI_Comm_get_parent(MPI_Comm * parent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_group:with:">PMPI_Comm_group: comm with: group	&lt;C: int __stdcall  PMPI_Comm_group(MPI_Comm comm, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_join:with:">PMPI_Comm_join: fd with: intercomm	&lt;C: int __stdcall  PMPI_Comm_join(int fd, MPI_Comm * intercomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_rank:with:">PMPI_Comm_rank: comm with: rank	&lt;C: int __stdcall  PMPI_Comm_rank(MPI_Comm comm, int * rank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_remote_group:with:">PMPI_Comm_remote_group: comm with: group	&lt;C: int __stdcall  PMPI_Comm_remote_group(MPI_Comm comm, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_remote_size:with:">PMPI_Comm_remote_size: comm with: size	&lt;C: int __stdcall  PMPI_Comm_remote_size(MPI_Comm comm, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_set_attr:with:with:">PMPI_Comm_set_attr: comm with: comm_keyval with: attribute_val	&lt;C: int __stdcall  PMPI_Comm_set_attr(MPI_Comm comm, int comm_keyval, void * attribute_val)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_set_errhandler:with:">PMPI_Comm_set_errhandler: comm with: errhandler	&lt;C: int __stdcall  PMPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_set_name:with:">PMPI_Comm_set_name: comm with: comm_name	&lt;C: int __stdcall  PMPI_Comm_set_name(MPI_Comm comm, const char * comm_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_size:with:">PMPI_Comm_size: comm with: size	&lt;C: int __stdcall  PMPI_Comm_size(MPI_Comm comm, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_spawn_multiple:with:with:with:with:with:with:with:with:">PMPI_Comm_spawn_multiple: count with: array_of_commands with: array_of_argv with: array_of_maxprocs with: array_of_info with: root with: comm with: intercomm with: array_of_errcodes	&lt;C: int __stdcall  PMPI_Comm_spawn_multiple(int count, char * array_of_commands[], char * * array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm * intercomm, int array_of_errcodes[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_split:with:with:with:">PMPI_Comm_split: comm with: color with: key with: newcomm	&lt;C: int __stdcall  PMPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_split_type:with:with:with:with:">PMPI_Comm_split_type: comm with: split_type with: key with: info with: newcomm	&lt;C: int __stdcall  PMPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm * newcomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Comm_test_inter:with:">PMPI_Comm_test_inter: comm with: flag	&lt;C: int __stdcall  PMPI_Comm_test_inter(MPI_Comm comm, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Compare_and_swap:with:with:with:with:with:with:">PMPI_Compare_and_swap: origin_addr with: compare_addr with: result_addr with: datatype with: target_rank with: target_disp with: win	&lt;C: int __stdcall  PMPI_Compare_and_swap(const void * origin_addr, const void * compare_addr, void * result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Dims_create:with:with:">PMPI_Dims_create: nnodes with: ndims with: dims	&lt;C: int __stdcall  PMPI_Dims_create(int nnodes, int ndims, int dims[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Dist_graph_create:with:with:with:with:with:with:with:with:">PMPI_Dist_graph_create: comm_old with: n with: sources with: degrees with: destinations with: weights with: info with: reorder with: comm_dist_graph	&lt;C: int __stdcall  PMPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[], const int degrees[], const int destinations[], const int weights[], MPI_Info info, int reorder, MPI_Comm * comm_dist_graph)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Dist_graph_create_adjacent:with:with:with:with:with:with:with:with:with:">PMPI_Dist_graph_create_adjacent: comm_old with: indegree with: sources with: sourceweights with: outdegree with: destinations with: destweights with: info with: reorder with: comm_dist_graph	&lt;C: int __stdcall  PMPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm * comm_dist_graph)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Dist_graph_neighbors:with:with:with:with:with:with:">PMPI_Dist_graph_neighbors: comm with: maxindegree with: sources with: sourceweights with: maxoutdegree with: destinations with: destweights	&lt;C: int __stdcall  PMPI_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], int sourceweights[], int maxoutdegree, int destinations[], int destweights[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Dist_graph_neighbors_count:with:with:with:">PMPI_Dist_graph_neighbors_count: comm with: indegree with: outdegree with: weighted	&lt;C: int __stdcall  PMPI_Dist_graph_neighbors_count(MPI_Comm comm, int * indegree, int * outdegree, int * weighted)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Errhandler_free:">PMPI_Errhandler_free: errhandler	&lt;C: int __stdcall  PMPI_Errhandler_free(MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Error_class:with:">PMPI_Error_class: errorcode with: errorclass	&lt;C: int __stdcall  PMPI_Error_class(int errorcode, int * errorclass)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Error_string:with:with:">PMPI_Error_string: errorcode with: string with: resultlen	&lt;C: int __stdcall  PMPI_Error_string(int errorcode, char * string, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Exscan:with:with:with:with:with:">PMPI_Exscan: sendbuf with: recvbuf with: count with: datatype with: op with: comm	&lt;C: int __stdcall  PMPI_Exscan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Fetch_and_op:with:with:with:with:with:with:">PMPI_Fetch_and_op: origin_addr with: result_addr with: datatype with: target_rank with: target_disp with: op with: win	&lt;C: int __stdcall  PMPI_Fetch_and_op(const void * origin_addr, void * result_addr, MPI_Datatype datatype, int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_call_errhandler:with:">PMPI_File_call_errhandler: file with: errorcode	&lt;C: int __stdcall  PMPI_File_call_errhandler(MPI_File file, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_close:">PMPI_File_close: fh	&lt;C: int __stdcall  PMPI_File_close(MPI_File * fh)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_create_errhandler:with:">PMPI_File_create_errhandler: function with: errhandler	&lt;C: int __stdcall  PMPI_File_create_errhandler(void ( __stdcall * function)(MPI_File * file, int * errcode, ...), MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_delete:with:">PMPI_File_delete: filename with: info	&lt;C: int __stdcall  PMPI_File_delete(const char * filename, MPI_Info info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_amode:with:">PMPI_File_get_amode: fh with: amode	&lt;C: int __stdcall  PMPI_File_get_amode(MPI_File fh, int * amode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_atomicity:with:">PMPI_File_get_atomicity: fh with: flag	&lt;C: int __stdcall  PMPI_File_get_atomicity(MPI_File fh, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_byte_offset:with:with:">PMPI_File_get_byte_offset: fh with: offset with: disp	&lt;C: int __stdcall  PMPI_File_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset * disp)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_errhandler:with:">PMPI_File_get_errhandler: file with: errhandler	&lt;C: int __stdcall  PMPI_File_get_errhandler(MPI_File file, MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_group:with:">PMPI_File_get_group: fh with: group	&lt;C: int __stdcall  PMPI_File_get_group(MPI_File fh, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_info:with:">PMPI_File_get_info: fh with: info_used	&lt;C: int __stdcall  PMPI_File_get_info(MPI_File fh, MPI_Info * info_used)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_position:with:">PMPI_File_get_position: fh with: offset	&lt;C: int __stdcall  PMPI_File_get_position(MPI_File fh, MPI_Offset * offset)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_position_shared:with:">PMPI_File_get_position_shared: fh with: offset	&lt;C: int __stdcall  PMPI_File_get_position_shared(MPI_File fh, MPI_Offset * offset)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_size:with:">PMPI_File_get_size: fh with: size	&lt;C: int __stdcall  PMPI_File_get_size(MPI_File fh, MPI_Offset * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_type_extent:with:with:">PMPI_File_get_type_extent: fh with: datatype with: extent	&lt;C: int __stdcall  PMPI_File_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint * extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_get_view:with:with:with:with:">PMPI_File_get_view: fh with: disp with: etype with: filetype with: datarep	&lt;C: int __stdcall  PMPI_File_get_view(MPI_File fh, MPI_Offset * disp, MPI_Datatype * etype, MPI_Datatype * filetype, char * datarep)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_iread:with:with:with:with:">PMPI_File_iread: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  PMPI_File_iread(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_iread_at:with:with:with:with:with:">PMPI_File_iread_at: fh with: offset with: buf with: count with: datatype with: request	&lt;C: int __stdcall  PMPI_File_iread_at(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_iread_shared:with:with:with:with:">PMPI_File_iread_shared: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  PMPI_File_iread_shared(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_iwrite:with:with:with:with:">PMPI_File_iwrite: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  PMPI_File_iwrite(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_iwrite_at:with:with:with:with:with:">PMPI_File_iwrite_at: fh with: offset with: buf with: count with: datatype with: request	&lt;C: int __stdcall  PMPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_iwrite_shared:with:with:with:with:">PMPI_File_iwrite_shared: fh with: buf with: count with: datatype with: request	&lt;C: int __stdcall  PMPI_File_iwrite_shared(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_open:with:with:with:with:">PMPI_File_open: comm with: filename with: amode with: info with: fh	&lt;C: int __stdcall  PMPI_File_open(MPI_Comm comm, const char * filename, int amode, MPI_Info info, MPI_File * fh)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_preallocate:with:">PMPI_File_preallocate: fh with: size	&lt;C: int __stdcall  PMPI_File_preallocate(MPI_File fh, MPI_Offset size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read:with:with:with:with:">PMPI_File_read: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_read(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_all:with:with:with:with:">PMPI_File_read_all: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_read_all(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_all_begin:with:with:with:">PMPI_File_read_all_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  PMPI_File_read_all_begin(MPI_File fh, void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_all_end:with:with:">PMPI_File_read_all_end: fh with: buf with: status	&lt;C: int __stdcall  PMPI_File_read_all_end(MPI_File fh, void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_at:with:with:with:with:with:">PMPI_File_read_at: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_read_at(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_at_all:with:with:with:with:with:">PMPI_File_read_at_all: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_read_at_all(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_at_all_begin:with:with:with:with:">PMPI_File_read_at_all_begin: fh with: offset with: buf with: count with: datatype	&lt;C: int __stdcall  PMPI_File_read_at_all_begin(MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_at_all_end:with:with:">PMPI_File_read_at_all_end: fh with: buf with: status	&lt;C: int __stdcall  PMPI_File_read_at_all_end(MPI_File fh, void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_ordered:with:with:with:with:">PMPI_File_read_ordered: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_read_ordered(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_ordered_begin:with:with:with:">PMPI_File_read_ordered_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  PMPI_File_read_ordered_begin(MPI_File fh, void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_ordered_end:with:with:">PMPI_File_read_ordered_end: fh with: buf with: status	&lt;C: int __stdcall  PMPI_File_read_ordered_end(MPI_File fh, void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_read_shared:with:with:with:with:">PMPI_File_read_shared: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_read_shared(MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_seek:with:with:">PMPI_File_seek: fh with: offset with: whence	&lt;C: int __stdcall  PMPI_File_seek(MPI_File fh, MPI_Offset offset, int whence)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_seek_shared:with:with:">PMPI_File_seek_shared: fh with: offset with: whence	&lt;C: int __stdcall  PMPI_File_seek_shared(MPI_File fh, MPI_Offset offset, int whence)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_set_atomicity:with:">PMPI_File_set_atomicity: fh with: flag	&lt;C: int __stdcall  PMPI_File_set_atomicity(MPI_File fh, int flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_set_errhandler:with:">PMPI_File_set_errhandler: file with: errhandler	&lt;C: int __stdcall  PMPI_File_set_errhandler(MPI_File file, MPI_Errhandler errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_set_info:with:">PMPI_File_set_info: fh with: info	&lt;C: int __stdcall  PMPI_File_set_info(MPI_File fh, MPI_Info info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_set_size:with:">PMPI_File_set_size: fh with: size	&lt;C: int __stdcall  PMPI_File_set_size(MPI_File fh, MPI_Offset size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_set_view:with:with:with:with:with:">PMPI_File_set_view: fh with: disp with: etype with: filetype with: datarep with: info	&lt;C: int __stdcall  PMPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char * datarep, MPI_Info info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_sync:">PMPI_File_sync: fh	&lt;C: int __stdcall  PMPI_File_sync(MPI_File fh)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write:with:with:with:with:">PMPI_File_write: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_write(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_all:with:with:with:with:">PMPI_File_write_all: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_write_all(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_all_begin:with:with:with:">PMPI_File_write_all_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  PMPI_File_write_all_begin(MPI_File fh, const void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_all_end:with:with:">PMPI_File_write_all_end: fh with: buf with: status	&lt;C: int __stdcall  PMPI_File_write_all_end(MPI_File fh, const void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_at:with:with:with:with:with:">PMPI_File_write_at: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_write_at(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_at_all:with:with:with:with:with:">PMPI_File_write_at_all: fh with: offset with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_write_at_all(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_at_all_begin:with:with:with:with:">PMPI_File_write_at_all_begin: fh with: offset with: buf with: count with: datatype	&lt;C: int __stdcall  PMPI_File_write_at_all_begin(MPI_File fh, MPI_Offset offset, const void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_at_all_end:with:with:">PMPI_File_write_at_all_end: fh with: buf with: status	&lt;C: int __stdcall  PMPI_File_write_at_all_end(MPI_File fh, const void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_ordered:with:with:with:with:">PMPI_File_write_ordered: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_write_ordered(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_ordered_begin:with:with:with:">PMPI_File_write_ordered_begin: fh with: buf with: count with: datatype	&lt;C: int __stdcall  PMPI_File_write_ordered_begin(MPI_File fh, const void * buf, int count, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_ordered_end:with:with:">PMPI_File_write_ordered_end: fh with: buf with: status	&lt;C: int __stdcall  PMPI_File_write_ordered_end(MPI_File fh, const void * buf, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_File_write_shared:with:with:with:with:">PMPI_File_write_shared: fh with: buf with: count with: datatype with: status	&lt;C: int __stdcall  PMPI_File_write_shared(MPI_File fh, const void * buf, int count, MPI_Datatype datatype, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Finalize">PMPI_Finalize	&lt;C: int __stdcall  PMPI_Finalize(void)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Finalized:">PMPI_Finalized: flag	&lt;C: int __stdcall  PMPI_Finalized(int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Free_mem:">PMPI_Free_mem: base	&lt;C: int __stdcall  PMPI_Free_mem(void * base)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Gather:with:with:with:with:with:with:with:">PMPI_Gather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm	&lt;C: int __stdcall  PMPI_Gather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Gatherv:with:with:with:with:with:with:with:with:">PMPI_Gatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: root with: comm	&lt;C: int __stdcall  PMPI_Gatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get:with:with:with:with:with:with:with:">PMPI_Get: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win	&lt;C: int __stdcall  PMPI_Get(void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_accumulate:with:with:with:with:with:with:with:with:with:with:with:">PMPI_Get_accumulate: origin_addr with: origin_count with: origin_datatype with: result_addr with: result_count with: result_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win	&lt;C: int __stdcall  PMPI_Get_accumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, void * result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_address:with:">PMPI_Get_address: location with: address	&lt;C: int __stdcall  PMPI_Get_address(const void * location, MPI_Aint * address)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_count:with:with:">PMPI_Get_count: status with: datatype with: count	&lt;C: int __stdcall  PMPI_Get_count(const MPI_Status * status, MPI_Datatype datatype, int * count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_elements:with:with:">PMPI_Get_elements: status with: datatype with: count	&lt;C: int __stdcall  PMPI_Get_elements(const MPI_Status * status, MPI_Datatype datatype, int * count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_elements_x:with:with:">PMPI_Get_elements_x: status with: datatype with: count	&lt;C: int __stdcall  PMPI_Get_elements_x(const MPI_Status * status, MPI_Datatype datatype, MPI_Count * count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_library_version:with:">PMPI_Get_library_version: version with: resultlen	&lt;C: int __stdcall  PMPI_Get_library_version(char * version, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_processor_name:with:">PMPI_Get_processor_name: name with: resultlen	&lt;C: int __stdcall  PMPI_Get_processor_name(char * name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Get_version:with:">PMPI_Get_version: version with: subversion	&lt;C: int __stdcall  PMPI_Get_version(int * version, int * subversion)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Graph_create:with:with:with:with:with:">PMPI_Graph_create: comm_old with: nnodes with: index with: edges with: reorder with: comm_graph	&lt;C: int __stdcall  PMPI_Graph_create(MPI_Comm comm_old, int nnodes, const int index[], const int edges[], int reorder, MPI_Comm * comm_graph)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Graph_get:with:with:with:with:">PMPI_Graph_get: comm with: maxindex with: maxedges with: index with: edges	&lt;C: int __stdcall  PMPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int index[], int edges[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Graph_map:with:with:with:with:">PMPI_Graph_map: comm with: nnodes with: index with: edges with: newrank	&lt;C: int __stdcall  PMPI_Graph_map(MPI_Comm comm, int nnodes, const int index[], const int edges[], int * newrank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Graph_neighbors:with:with:with:">PMPI_Graph_neighbors: comm with: rank with: maxneighbors with: neighbors	&lt;C: int __stdcall  PMPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Graph_neighbors_count:with:with:">PMPI_Graph_neighbors_count: comm with: rank with: nneighbors	&lt;C: int __stdcall  PMPI_Graph_neighbors_count(MPI_Comm comm, int rank, int * nneighbors)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Graphdims_get:with:with:">PMPI_Graphdims_get: comm with: nnodes with: nedges	&lt;C: int __stdcall  PMPI_Graphdims_get(MPI_Comm comm, int * nnodes, int * nedges)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Grequest_complete:">PMPI_Grequest_complete: request	&lt;C: int __stdcall  PMPI_Grequest_complete(MPI_Request request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Grequest_start:with:with:with:with:">PMPI_Grequest_start: query_fn with: free_fn with: cancel_fn with: extra_state with: request	&lt;C: int __stdcall  PMPI_Grequest_start(int ( __stdcall * query_fn)(void * extra_state, MPI_Status * status), int ( __stdcall * free_fn)(void * extra_state), int ( __stdcall * cancel_fn)(void * extra_state, int complete), void * extra_state, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_compare:with:with:">PMPI_Group_compare: group1 with: group2 with: result	&lt;C: int __stdcall  PMPI_Group_compare(MPI_Group group1, MPI_Group group2, int * result)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_difference:with:with:">PMPI_Group_difference: group1 with: group2 with: newgroup	&lt;C: int __stdcall  PMPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_excl:with:with:with:">PMPI_Group_excl: group with: n with: ranks with: newgroup	&lt;C: int __stdcall  PMPI_Group_excl(MPI_Group group, int n, const int ranks[], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_free:">PMPI_Group_free: group	&lt;C: int __stdcall  PMPI_Group_free(MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_incl:with:with:with:">PMPI_Group_incl: group with: n with: ranks with: newgroup	&lt;C: int __stdcall  PMPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_intersection:with:with:">PMPI_Group_intersection: group1 with: group2 with: newgroup	&lt;C: int __stdcall  PMPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_range_excl:with:with:with:">PMPI_Group_range_excl: group with: n with: ranges with: newgroup	&lt;C: int __stdcall  PMPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_range_incl:with:with:with:">PMPI_Group_range_incl: group with: n with: ranges with: newgroup	&lt;C: int __stdcall  PMPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_rank:with:">PMPI_Group_rank: group with: rank	&lt;C: int __stdcall  PMPI_Group_rank(MPI_Group group, int * rank)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_size:with:">PMPI_Group_size: group with: size	&lt;C: int __stdcall  PMPI_Group_size(MPI_Group group, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_translate_ranks:with:with:with:with:">PMPI_Group_translate_ranks: group1 with: n with: ranks1 with: group2 with: ranks2	&lt;C: int __stdcall  PMPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Group_union:with:with:">PMPI_Group_union: group1 with: group2 with: newgroup	&lt;C: int __stdcall  PMPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group * newgroup)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iallgather:with:with:with:with:with:with:with:">PMPI_Iallgather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm with: request	&lt;C: int __stdcall  PMPI_Iallgather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iallgatherv:with:with:with:with:with:with:with:with:">PMPI_Iallgatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: comm with: request	&lt;C: int __stdcall  PMPI_Iallgatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iallreduce:with:with:with:with:with:with:">PMPI_Iallreduce: sendbuf with: recvbuf with: count with: datatype with: op with: comm with: request	&lt;C: int __stdcall  PMPI_Iallreduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ialltoall:with:with:with:with:with:with:with:">PMPI_Ialltoall: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: comm with: request	&lt;C: int __stdcall  PMPI_Ialltoall(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ialltoallv:with:with:with:with:with:with:with:with:with:">PMPI_Ialltoallv: sendbuf with: sendcounts with: sdispls with: sendtype with: recvbuf with: recvcounts with: rdispls with: recvtype with: comm with: request	&lt;C: int __stdcall  PMPI_Ialltoallv(const void * sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ialltoallw:with:with:with:with:with:with:with:with:with:">PMPI_Ialltoallw: sendbuf with: sendcounts with: sdispls with: sendtypes with: recvbuf with: recvcounts with: rdispls with: recvtypes with: comm with: request	&lt;C: int __stdcall  PMPI_Ialltoallw(const void * sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void * recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ibarrier:with:">PMPI_Ibarrier: comm with: request	&lt;C: int __stdcall  PMPI_Ibarrier(MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ibcast:with:with:with:with:with:">PMPI_Ibcast: buffer with: count with: datatype with: root with: comm with: request	&lt;C: int __stdcall  PMPI_Ibcast(void * buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ibsend:with:with:with:with:with:with:">PMPI_Ibsend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Ibsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iexscan:with:with:with:with:with:with:">PMPI_Iexscan: sendbuf with: recvbuf with: count with: datatype with: op with: comm with: request	&lt;C: int __stdcall  PMPI_Iexscan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Igather:with:with:with:with:with:with:with:with:">PMPI_Igather: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  PMPI_Igather(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Igatherv:with:with:with:with:with:with:with:with:with:">PMPI_Igatherv: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcounts with: displs with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  PMPI_Igatherv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Improbe:with:with:with:with:with:">PMPI_Improbe: source with: tag with: comm with: flag with: message with: status	&lt;C: int __stdcall  PMPI_Improbe(int source, int tag, MPI_Comm comm, int * flag, MPI_Message * message, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Imrecv:with:with:with:with:">PMPI_Imrecv: buf with: count with: datatype with: message with: request	&lt;C: int __stdcall  PMPI_Imrecv(void * buf, int count, MPI_Datatype datatype, MPI_Message * message, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_create:">PMPI_Info_create: info	&lt;C: int __stdcall  PMPI_Info_create(MPI_Info * info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_delete:with:">PMPI_Info_delete: info with: key	&lt;C: int __stdcall  PMPI_Info_delete(MPI_Info info, const char * key)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_dup:with:">PMPI_Info_dup: info with: newinfo	&lt;C: int __stdcall  PMPI_Info_dup(MPI_Info info, MPI_Info * newinfo)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_free:">PMPI_Info_free: info	&lt;C: int __stdcall  PMPI_Info_free(MPI_Info * info)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_get:with:with:with:with:">PMPI_Info_get: info with: key with: valuelen with: value with: flag	&lt;C: int __stdcall  PMPI_Info_get(MPI_Info info, const char * key, int valuelen, char * value, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_get_nkeys:with:">PMPI_Info_get_nkeys: info with: nkeys	&lt;C: int __stdcall  PMPI_Info_get_nkeys(MPI_Info info, int * nkeys)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_get_nthkey:with:with:">PMPI_Info_get_nthkey: info with: n with: key	&lt;C: int __stdcall  PMPI_Info_get_nthkey(MPI_Info info, int n, char * key)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_get_valuelen:with:with:with:">PMPI_Info_get_valuelen: info with: key with: valuelen with: flag	&lt;C: int __stdcall  PMPI_Info_get_valuelen(MPI_Info info, const char * key, int * valuelen, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Info_set:with:with:">PMPI_Info_set: info with: key with: value	&lt;C: int __stdcall  PMPI_Info_set(MPI_Info info, const char * key, const char * value)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Init:with:">PMPI_Init: argc with: argv	&lt;C: int __stdcall  PMPI_Init(int * argc, char * * * argv)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Init_thread:with:with:with:">PMPI_Init_thread: argc with: argv with: required with: provided	&lt;C: int __stdcall  PMPI_Init_thread(int * argc, char * * * argv, int required, int * provided)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Initialized:">PMPI_Initialized: flag	&lt;C: int __stdcall  PMPI_Initialized(int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Intercomm_create:with:with:with:with:with:">PMPI_Intercomm_create: local_comm with: local_leader with: peer_comm with: remote_leader with: tag with: newintercomm	&lt;C: int __stdcall  PMPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm * newintercomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Intercomm_merge:with:with:">PMPI_Intercomm_merge: intercomm with: high with: newintracomm	&lt;C: int __stdcall  PMPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm * newintracomm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iprobe:with:with:with:with:">PMPI_Iprobe: source with: tag with: comm with: flag with: status	&lt;C: int __stdcall  PMPI_Iprobe(int source, int tag, MPI_Comm comm, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Irecv:with:with:with:with:with:with:">PMPI_Irecv: buf with: count with: datatype with: source with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Irecv(void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ireduce:with:with:with:with:with:with:with:">PMPI_Ireduce: sendbuf with: recvbuf with: count with: datatype with: op with: root with: comm with: request	&lt;C: int __stdcall  PMPI_Ireduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ireduce_scatter:with:with:with:with:with:with:">PMPI_Ireduce_scatter: sendbuf with: recvbuf with: recvcounts with: datatype with: op with: comm with: request	&lt;C: int __stdcall  PMPI_Ireduce_scatter(const void * sendbuf, void * recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ireduce_scatter_block:with:with:with:with:with:with:">PMPI_Ireduce_scatter_block: sendbuf with: recvbuf with: recvcount with: datatype with: op with: comm with: request	&lt;C: int __stdcall  PMPI_Ireduce_scatter_block(const void * sendbuf, void * recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Irsend:with:with:with:with:with:with:">PMPI_Irsend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Irsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Is_thread_main:">PMPI_Is_thread_main: flag	&lt;C: int __stdcall  PMPI_Is_thread_main(int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iscan:with:with:with:with:with:with:">PMPI_Iscan: sendbuf with: recvbuf with: count with: datatype with: op with: comm with: request	&lt;C: int __stdcall  PMPI_Iscan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iscatter:with:with:with:with:with:with:with:with:">PMPI_Iscatter: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  PMPI_Iscatter(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Iscatterv:with:with:with:with:with:with:with:with:with:">PMPI_Iscatterv: sendbuf with: sendcounts with: displs with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm with: request	&lt;C: int __stdcall  PMPI_Iscatterv(const void * sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Isend:with:with:with:with:with:with:">PMPI_Isend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Isend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Issend:with:with:with:with:with:with:">PMPI_Issend: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Issend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Lookup_name:with:with:">PMPI_Lookup_name: service_name with: info with: port_name	&lt;C: int __stdcall  PMPI_Lookup_name(const char * service_name, MPI_Info info, char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Mprobe:with:with:with:with:">PMPI_Mprobe: source with: tag with: comm with: message with: status	&lt;C: int __stdcall  PMPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message * message, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Mrecv:with:with:with:with:">PMPI_Mrecv: buf with: count with: datatype with: message with: status	&lt;C: int __stdcall  PMPI_Mrecv(void * buf, int count, MPI_Datatype datatype, MPI_Message * message, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Op_commutative:with:">PMPI_Op_commutative: op with: commute	&lt;C: int __stdcall  PMPI_Op_commutative(MPI_Op op, int * commute)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Op_create:with:with:">PMPI_Op_create: user_fn with: commute with: op	&lt;C: int __stdcall  PMPI_Op_create(void ( __stdcall * user_fn)(void * invec, void * inoutvec, int * len, MPI_Datatype * datatype), int commute, MPI_Op * op)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Op_free:">PMPI_Op_free: op	&lt;C: int __stdcall  PMPI_Op_free(MPI_Op * op)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Open_port:with:">PMPI_Open_port: info with: port_name	&lt;C: int __stdcall  PMPI_Open_port(MPI_Info info, char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Pack:with:with:with:with:with:with:">PMPI_Pack: inbuf with: incount with: datatype with: outbuf with: outsize with: position with: comm	&lt;C: int __stdcall  PMPI_Pack(const void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, int outsize, int * position, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Pack_external:with:with:with:with:with:with:">PMPI_Pack_external: datarep with: inbuf with: incount with: datatype with: outbuf with: outsize with: position	&lt;C: int __stdcall  PMPI_Pack_external(const char * datarep, const void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, MPI_Aint outsize, MPI_Aint * position)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Pack_external_size:with:with:with:">PMPI_Pack_external_size: datarep with: incount with: datatype with: size	&lt;C: int __stdcall  PMPI_Pack_external_size(const char * datarep, int incount, MPI_Datatype datatype, MPI_Aint * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Pack_size:with:with:with:">PMPI_Pack_size: incount with: datatype with: comm with: size	&lt;C: int __stdcall  PMPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Probe:with:with:with:">PMPI_Probe: source with: tag with: comm with: status	&lt;C: int __stdcall  PMPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Publish_name:with:with:">PMPI_Publish_name: service_name with: info with: port_name	&lt;C: int __stdcall  PMPI_Publish_name(const char * service_name, MPI_Info info, const char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Put:with:with:with:with:with:with:with:">PMPI_Put: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win	&lt;C: int __stdcall  PMPI_Put(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Query_thread:">PMPI_Query_thread: provided	&lt;C: int __stdcall  PMPI_Query_thread(int * provided)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Raccumulate:with:with:with:with:with:with:with:with:with:">PMPI_Raccumulate: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win with: request	&lt;C: int __stdcall  PMPI_Raccumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Recv:with:with:with:with:with:with:">PMPI_Recv: buf with: count with: datatype with: source with: tag with: comm with: status	&lt;C: int __stdcall  PMPI_Recv(void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Recv_init:with:with:with:with:with:with:">PMPI_Recv_init: buf with: count with: datatype with: source with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Recv_init(void * buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Reduce:with:with:with:with:with:with:">PMPI_Reduce: sendbuf with: recvbuf with: count with: datatype with: op with: root with: comm	&lt;C: int __stdcall  PMPI_Reduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Reduce_local:with:with:with:with:">PMPI_Reduce_local: inbuf with: inoutbuf with: count with: datatype with: op	&lt;C: int __stdcall  PMPI_Reduce_local(const void * inbuf, void * inoutbuf, int count, MPI_Datatype datatype, MPI_Op op)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Reduce_scatter:with:with:with:with:with:">PMPI_Reduce_scatter: sendbuf with: recvbuf with: recvcounts with: datatype with: op with: comm	&lt;C: int __stdcall  PMPI_Reduce_scatter(const void * sendbuf, void * recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Reduce_scatter_block:with:with:with:with:with:">PMPI_Reduce_scatter_block: sendbuf with: recvbuf with: recvcount with: datatype with: op with: comm	&lt;C: int __stdcall  PMPI_Reduce_scatter_block(const void * sendbuf, void * recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Register_datarep:with:with:with:with:">PMPI_Register_datarep: datarep with: read_conversion_fn with: write_conversion_fn with: dtype_file_extent_fn with: extra_state	&lt;C: int __stdcall  PMPI_Register_datarep(const char * datarep, int ( __stdcall * read_conversion_fn)(void * userbuf, MPI_Datatype datatype, int count, void * filebuf, MPI_Offset position, void * extra_state), int ( __stdcall * write_conversion_fn)(void * userbuf, MPI_Datatype datatype, int count, void * filebuf, MPI_Offset position, void * extra_state), int ( __stdcall * dtype_file_extent_fn)(MPI_Datatype datatype, MPI_Aint * file_extent, void * extra_state), void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Request_free:">PMPI_Request_free: request	&lt;C: int __stdcall  PMPI_Request_free(MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Request_get_status:with:with:">PMPI_Request_get_status: request with: flag with: status	&lt;C: int __stdcall  PMPI_Request_get_status(MPI_Request request, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Rget:with:with:with:with:with:with:with:with:">PMPI_Rget: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win with: request	&lt;C: int __stdcall  PMPI_Rget(void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Rget_accumulate:with:with:with:with:with:with:with:with:with:with:with:with:">PMPI_Rget_accumulate: origin_addr with: origin_count with: origin_datatype with: result_addr with: result_count with: result_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: op with: win with: request	&lt;C: int __stdcall  PMPI_Rget_accumulate(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, void * result_addr, int result_count, MPI_Datatype result_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Rput:with:with:with:with:with:with:with:with:">PMPI_Rput: origin_addr with: origin_count with: origin_datatype with: target_rank with: target_disp with: target_count with: target_datatype with: win with: request	&lt;C: int __stdcall  PMPI_Rput(const void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Rsend:with:with:with:with:with:">PMPI_Rsend: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  PMPI_Rsend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Rsend_init:with:with:with:with:with:with:">PMPI_Rsend_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Rsend_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Scan:with:with:with:with:with:">PMPI_Scan: sendbuf with: recvbuf with: count with: datatype with: op with: comm	&lt;C: int __stdcall  PMPI_Scan(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Scatter:with:with:with:with:with:with:with:">PMPI_Scatter: sendbuf with: sendcount with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm	&lt;C: int __stdcall  PMPI_Scatter(const void * sendbuf, int sendcount, MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Scatterv:with:with:with:with:with:with:with:with:">PMPI_Scatterv: sendbuf with: sendcounts with: displs with: sendtype with: recvbuf with: recvcount with: recvtype with: root with: comm	&lt;C: int __stdcall  PMPI_Scatterv(const void * sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void * recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Send:with:with:with:with:with:">PMPI_Send: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  PMPI_Send(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Send_init:with:with:with:with:with:with:">PMPI_Send_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Send_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Sendrecv:with:with:with:with:with:with:with:with:with:with:with:">PMPI_Sendrecv: sendbuf with: sendcount with: sendtype with: dest with: sendtag with: recvbuf with: recvcount with: recvtype with: source with: recvtag with: comm with: status	&lt;C: int __stdcall  PMPI_Sendrecv(const void * sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void * recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Sendrecv_replace:with:with:with:with:with:with:with:with:">PMPI_Sendrecv_replace: buf with: count with: datatype with: dest with: sendtag with: source with: recvtag with: comm with: status	&lt;C: int __stdcall  PMPI_Sendrecv_replace(void * buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ssend:with:with:with:with:with:">PMPI_Ssend: buf with: count with: datatype with: dest with: tag with: comm	&lt;C: int __stdcall  PMPI_Ssend(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Ssend_init:with:with:with:with:with:with:">PMPI_Ssend_init: buf with: count with: datatype with: dest with: tag with: comm with: request	&lt;C: int __stdcall  PMPI_Ssend_init(const void * buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Start:">PMPI_Start: request	&lt;C: int __stdcall  PMPI_Start(MPI_Request * request)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Startall:with:">PMPI_Startall: count with: array_of_requests	&lt;C: int __stdcall  PMPI_Startall(int count, MPI_Request array_of_requests[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Status_set_cancelled:with:">PMPI_Status_set_cancelled: status with: flag	&lt;C: int __stdcall  PMPI_Status_set_cancelled(MPI_Status * status, int flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Status_set_elements:with:with:">PMPI_Status_set_elements: status with: datatype with: count	&lt;C: int __stdcall  PMPI_Status_set_elements(MPI_Status * status, MPI_Datatype datatype, int count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Status_set_elements_x:with:with:">PMPI_Status_set_elements_x: status with: datatype with: count	&lt;C: int __stdcall  PMPI_Status_set_elements_x(MPI_Status * status, MPI_Datatype datatype, MPI_Count count)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Test:with:with:">PMPI_Test: request with: flag with: status	&lt;C: int __stdcall  PMPI_Test(MPI_Request * request, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Test_cancelled:with:">PMPI_Test_cancelled: status with: flag	&lt;C: int __stdcall  PMPI_Test_cancelled(const MPI_Status * status, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Testall:with:with:with:">PMPI_Testall: count with: array_of_requests with: flag with: array_of_statuses	&lt;C: int __stdcall  PMPI_Testall(int count, MPI_Request array_of_requests[], int * flag, MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Testany:with:with:with:with:">PMPI_Testany: count with: array_of_requests with: index with: flag with: status	&lt;C: int __stdcall  PMPI_Testany(int count, MPI_Request array_of_requests[], int * index, int * flag, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Testsome:with:with:with:with:">PMPI_Testsome: incount with: array_of_requests with: outcount with: array_of_indices with: array_of_statuses	&lt;C: int __stdcall  PMPI_Testsome(int incount, MPI_Request array_of_requests[], int * outcount, int array_of_indices[], MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Topo_test:with:">PMPI_Topo_test: comm with: status	&lt;C: int __stdcall  PMPI_Topo_test(MPI_Comm comm, int * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_commit:">PMPI_Type_commit: datatype	&lt;C: int __stdcall  PMPI_Type_commit(MPI_Datatype * datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_contiguous:with:with:">PMPI_Type_contiguous: count with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_darray:with:with:with:with:with:with:with:with:with:">PMPI_Type_create_darray: size with: rank with: ndims with: array_of_gsizes with: array_of_distribs with: array_of_dargs with: array_of_psizes with: order with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_create_darray(int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[], const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_hindexed:with:with:with:with:">PMPI_Type_create_hindexed: count with: array_of_blocklengths with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_create_hindexed(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_hindexed_block:with:with:with:with:">PMPI_Type_create_hindexed_block: count with: blocklength with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_create_hindexed_block(int count, int blocklength, const MPI_Aint array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_hvector:with:with:with:with:">PMPI_Type_create_hvector: count with: blocklength with: stride with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_create_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_indexed_block:with:with:with:with:">PMPI_Type_create_indexed_block: count with: blocklength with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_create_indexed_block(int count, int blocklength, const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_keyval:with:with:with:">PMPI_Type_create_keyval: type_copy_attr_fn with: type_delete_attr_fn with: type_keyval with: extra_state	&lt;C: int __stdcall  PMPI_Type_create_keyval(int ( __stdcall * type_copy_attr_fn)(MPI_Datatype olddatatype, int datatype_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag), int ( __stdcall * type_delete_attr_fn)(MPI_Datatype datatype, int datatype_keyval, void * attribute_val, void * extra_state), int * type_keyval, void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_resized:with:with:with:">PMPI_Type_create_resized: oldtype with: lb with: extent with: newtype	&lt;C: int __stdcall  PMPI_Type_create_resized(MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_struct:with:with:with:with:">PMPI_Type_create_struct: count with: array_of_blocklengths with: array_of_displacements with: array_of_types with: newtype	&lt;C: int __stdcall  PMPI_Type_create_struct(int count, const int array_of_blocklengths[], const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[], MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_create_subarray:with:with:with:with:with:with:">PMPI_Type_create_subarray: ndims with: array_of_sizes with: array_of_subsizes with: array_of_starts with: order with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_create_subarray(int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_delete_attr:with:">PMPI_Type_delete_attr: type with: type_keyval	&lt;C: int __stdcall  PMPI_Type_delete_attr(MPI_Datatype type, int type_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_dup:with:">PMPI_Type_dup: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_free:">PMPI_Type_free: datatype	&lt;C: int __stdcall  PMPI_Type_free(MPI_Datatype * datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_free_keyval:">PMPI_Type_free_keyval: type_keyval	&lt;C: int __stdcall  PMPI_Type_free_keyval(int * type_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_attr:with:with:with:">PMPI_Type_get_attr: type with: type_keyval with: attribute_val with: flag	&lt;C: int __stdcall  PMPI_Type_get_attr(MPI_Datatype type, int type_keyval, void * attribute_val, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_contents:with:with:with:with:with:with:">PMPI_Type_get_contents: datatype with: max_integers with: max_addresses with: max_datatypes with: array_of_integers with: array_of_addresses with: array_of_datatypes	&lt;C: int __stdcall  PMPI_Type_get_contents(MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int array_of_integers[], MPI_Aint array_of_addresses[], MPI_Datatype array_of_datatypes[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_envelope:with:with:with:with:">PMPI_Type_get_envelope: datatype with: num_integers with: num_addresses with: num_datatypes with: combiner	&lt;C: int __stdcall  PMPI_Type_get_envelope(MPI_Datatype datatype, int * num_integers, int * num_addresses, int * num_datatypes, int * combiner)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_extent:with:with:">PMPI_Type_get_extent: datatype with: lb with: extent	&lt;C: int __stdcall  PMPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint * lb, MPI_Aint * extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_extent_x:with:with:">PMPI_Type_get_extent_x: datatype with: lb with: extent	&lt;C: int __stdcall  PMPI_Type_get_extent_x(MPI_Datatype datatype, MPI_Count * lb, MPI_Count * extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_name:with:with:">PMPI_Type_get_name: datatype with: type_name with: resultlen	&lt;C: int __stdcall  PMPI_Type_get_name(MPI_Datatype datatype, char * type_name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_true_extent:with:with:">PMPI_Type_get_true_extent: datatype with: true_lb with: true_extent	&lt;C: int __stdcall  PMPI_Type_get_true_extent(MPI_Datatype datatype, MPI_Aint * true_lb, MPI_Aint * true_extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_get_true_extent_x:with:with:">PMPI_Type_get_true_extent_x: datatype with: true_lb with: true_extent	&lt;C: int __stdcall  PMPI_Type_get_true_extent_x(MPI_Datatype datatype, MPI_Count * true_lb, MPI_Count * true_extent)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_indexed:with:with:with:with:">PMPI_Type_indexed: count with: array_of_blocklengths with: array_of_displacements with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_indexed(int count, const int array_of_blocklengths[], const int array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_set_attr:with:with:">PMPI_Type_set_attr: type with: type_keyval with: attribute_val	&lt;C: int __stdcall  PMPI_Type_set_attr(MPI_Datatype type, int type_keyval, void * attribute_val)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_set_name:with:">PMPI_Type_set_name: datatype with: type_name	&lt;C: int __stdcall  PMPI_Type_set_name(MPI_Datatype datatype, const char * type_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_size:with:">PMPI_Type_size: datatype with: size	&lt;C: int __stdcall  PMPI_Type_size(MPI_Datatype datatype, int * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_size_x:with:">PMPI_Type_size_x: datatype with: size	&lt;C: int __stdcall  PMPI_Type_size_x(MPI_Datatype datatype, MPI_Count * size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Type_vector:with:with:with:with:">PMPI_Type_vector: count with: blocklength with: stride with: oldtype with: newtype	&lt;C: int __stdcall  PMPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype * newtype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Unpack:with:with:with:with:with:with:">PMPI_Unpack: inbuf with: insize with: position with: outbuf with: outcount with: datatype with: comm	&lt;C: int __stdcall  PMPI_Unpack(const void * inbuf, int insize, int * position, void * outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Unpack_external:with:with:with:with:with:with:">PMPI_Unpack_external: datarep with: inbuf with: insize with: position with: outbuf with: outcount with: datatype	&lt;C: int __stdcall  PMPI_Unpack_external(const char * datarep, const void * inbuf, MPI_Aint insize, MPI_Aint * position, void * outbuf, int outcount, MPI_Datatype datatype)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Unpublish_name:with:with:">PMPI_Unpublish_name: service_name with: info with: port_name	&lt;C: int __stdcall  PMPI_Unpublish_name(const char * service_name, MPI_Info info, const char * port_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Wait:with:">PMPI_Wait: request with: status	&lt;C: int __stdcall  PMPI_Wait(MPI_Request * request, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Waitall:with:with:">PMPI_Waitall: count with: array_of_requests with: array_of_statuses	&lt;C: int __stdcall  PMPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Waitany:with:with:with:">PMPI_Waitany: count with: array_of_requests with: index with: status	&lt;C: int __stdcall  PMPI_Waitany(int count, MPI_Request array_of_requests[], int * index, MPI_Status * status)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Waitsome:with:with:with:with:">PMPI_Waitsome: incount with: array_of_requests with: outcount with: array_of_indices with: array_of_statuses	&lt;C: int __stdcall  PMPI_Waitsome(int incount, MPI_Request array_of_requests[], int * outcount, int array_of_indices[], MPI_Status array_of_statuses[])&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_allocate:with:with:with:with:with:">PMPI_Win_allocate: size with: disp_unit with: info with: comm with: baseptr with: win	&lt;C: int __stdcall  PMPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void * baseptr, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_allocate_shared:with:with:with:with:with:">PMPI_Win_allocate_shared: size with: disp_unit with: info with: comm with: baseptr with: win	&lt;C: int __stdcall  PMPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void * baseptr, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_attach:with:with:">PMPI_Win_attach: win with: base with: size	&lt;C: int __stdcall  PMPI_Win_attach(MPI_Win win, void * base, MPI_Aint size)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_call_errhandler:with:">PMPI_Win_call_errhandler: win with: errorcode	&lt;C: int __stdcall  PMPI_Win_call_errhandler(MPI_Win win, int errorcode)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_complete:">PMPI_Win_complete: win	&lt;C: int __stdcall  PMPI_Win_complete(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_create:with:with:with:with:with:">PMPI_Win_create: base with: size with: disp_unit with: info with: comm with: win	&lt;C: int __stdcall  PMPI_Win_create(void * base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_create_dynamic:with:with:">PMPI_Win_create_dynamic: info with: comm with: win	&lt;C: int __stdcall  PMPI_Win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_create_errhandler:with:">PMPI_Win_create_errhandler: function with: errhandler	&lt;C: int __stdcall  PMPI_Win_create_errhandler(void ( __stdcall * function)(MPI_Win * win, int * errcode, ...), MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_create_keyval:with:with:with:">PMPI_Win_create_keyval: win_copy_attr_fn with: win_delete_attr_fn with: win_keyval with: extra_state	&lt;C: int __stdcall  PMPI_Win_create_keyval(int ( __stdcall * win_copy_attr_fn)(MPI_Win oldwin, int win_keyval, void * extra_state, void * attribute_val_in, void * attribute_val_out, int * flag), int ( __stdcall * win_delete_attr_fn)(MPI_Win win, int win_keyval, void * attribute_val, void * extra_state), int * win_keyval, void * extra_state)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_delete_attr:with:">PMPI_Win_delete_attr: win with: win_keyval	&lt;C: int __stdcall  PMPI_Win_delete_attr(MPI_Win win, int win_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_detach:with:">PMPI_Win_detach: win with: base	&lt;C: int __stdcall  PMPI_Win_detach(MPI_Win win, void * base)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_fence:with:">PMPI_Win_fence: assert with: win	&lt;C: int __stdcall  PMPI_Win_fence(int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_flush:with:">PMPI_Win_flush: rank with: win	&lt;C: int __stdcall  PMPI_Win_flush(int rank, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_flush_all:">PMPI_Win_flush_all: win	&lt;C: int __stdcall  PMPI_Win_flush_all(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_flush_local:with:">PMPI_Win_flush_local: rank with: win	&lt;C: int __stdcall  PMPI_Win_flush_local(int rank, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_flush_local_all:">PMPI_Win_flush_local_all: win	&lt;C: int __stdcall  PMPI_Win_flush_local_all(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_free:">PMPI_Win_free: win	&lt;C: int __stdcall  PMPI_Win_free(MPI_Win * win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_free_keyval:">PMPI_Win_free_keyval: win_keyval	&lt;C: int __stdcall  PMPI_Win_free_keyval(int * win_keyval)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_get_attr:with:with:with:">PMPI_Win_get_attr: win with: win_keyval with: attribute_val with: flag	&lt;C: int __stdcall  PMPI_Win_get_attr(MPI_Win win, int win_keyval, void * attribute_val, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_get_errhandler:with:">PMPI_Win_get_errhandler: win with: errhandler	&lt;C: int __stdcall  PMPI_Win_get_errhandler(MPI_Win win, MPI_Errhandler * errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_get_group:with:">PMPI_Win_get_group: win with: group	&lt;C: int __stdcall  PMPI_Win_get_group(MPI_Win win, MPI_Group * group)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_get_name:with:with:">PMPI_Win_get_name: win with: win_name with: resultlen	&lt;C: int __stdcall  PMPI_Win_get_name(MPI_Win win, char * win_name, int * resultlen)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_lock:with:with:with:">PMPI_Win_lock: lock_type with: rank with: assert with: win	&lt;C: int __stdcall  PMPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_lock_all:with:">PMPI_Win_lock_all: assert with: win	&lt;C: int __stdcall  PMPI_Win_lock_all(int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_post:with:with:">PMPI_Win_post: group with: assert with: win	&lt;C: int __stdcall  PMPI_Win_post(MPI_Group group, int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_set_attr:with:with:">PMPI_Win_set_attr: win with: win_keyval with: attribute_val	&lt;C: int __stdcall  PMPI_Win_set_attr(MPI_Win win, int win_keyval, void * attribute_val)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_set_errhandler:with:">PMPI_Win_set_errhandler: win with: errhandler	&lt;C: int __stdcall  PMPI_Win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_set_name:with:">PMPI_Win_set_name: win with: win_name	&lt;C: int __stdcall  PMPI_Win_set_name(MPI_Win win, const char * win_name)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_shared_query:with:with:with:with:">PMPI_Win_shared_query: win with: rank with: size with: disp_unit with: baseptr	&lt;C: int __stdcall  PMPI_Win_shared_query(MPI_Win win, int rank, MPI_Aint * size, int * disp_unit, void * baseptr)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_start:with:with:">PMPI_Win_start: group with: assert with: win	&lt;C: int __stdcall  PMPI_Win_start(MPI_Group group, int assert, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_sync:">PMPI_Win_sync: win	&lt;C: int __stdcall  PMPI_Win_sync(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_test:with:">PMPI_Win_test: win with: flag	&lt;C: int __stdcall  PMPI_Win_test(MPI_Win win, int * flag)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_unlock:with:">PMPI_Win_unlock: rank with: win	&lt;C: int __stdcall  PMPI_Win_unlock(int rank, MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_unlock_all:">PMPI_Win_unlock_all: win	&lt;C: int __stdcall  PMPI_Win_unlock_all(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Win_wait:">PMPI_Win_wait: win	&lt;C: int __stdcall  PMPI_Win_wait(MPI_Win win)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Wtick">PMPI_Wtick	&lt;C: double __stdcall  PMPI_Wtick(void)&gt;	^self externalAccessFailedWith: _errorCode</body>

<body package="StMPI GUI" selector="PMPI_Wtime">PMPI_Wtime	&lt;C: double __stdcall  PMPI_Wtime(void)&gt;	^self externalAccessFailedWith: _errorCode</body>
</methods>

<methods>
<class-id>StMPI</class-id> <category>defines</category>

<body package="StMPI GUI" selector="MPIAPI">MPIAPI	&lt;C: #define MPIAPI __stdcall	&gt;</body>

<body package="StMPI GUI" selector="MPICH_ERR_LAST_CLASS">MPICH_ERR_LAST_CLASS	&lt;C: #define MPICH_ERR_LAST_CLASS 53	&gt;</body>

<body package="StMPI GUI" selector="MPIU_DLL_SPEC">MPIU_DLL_SPEC	&lt;C: #define MPIU_DLL_SPEC __declspec(dllimport)	&gt;</body>

<body package="StMPI GUI" selector="MPI_2COMPLEX">MPI_2COMPLEX	&lt;C: #define MPI_2COMPLEX ((MPI_Datatype)1275072548)	&gt;</body>

<body package="StMPI GUI" selector="MPI_2DOUBLE_COMPLEX">MPI_2DOUBLE_COMPLEX	&lt;C: #define MPI_2DOUBLE_COMPLEX ((MPI_Datatype)1275076645)	&gt;</body>

<body package="StMPI GUI" selector="MPI_2DOUBLE_PRECISION">MPI_2DOUBLE_PRECISION	&lt;C: #define MPI_2DOUBLE_PRECISION ((MPI_Datatype)1275072547)	&gt;</body>

<body package="StMPI GUI" selector="MPI_2INT">MPI_2INT	&lt;C: #define MPI_2INT ((MPI_Datatype)1275070486)	&gt;</body>

<body package="StMPI GUI" selector="MPI_2INTEGER">MPI_2INTEGER	&lt;C: #define MPI_2INTEGER ((MPI_Datatype)1275070496)	&gt;</body>

<body package="StMPI GUI" selector="MPI_2REAL">MPI_2REAL	&lt;C: #define MPI_2REAL ((MPI_Datatype)1275070497)	&gt;</body>

<body package="StMPI GUI" selector="MPI_AINT">MPI_AINT	&lt;C: #define MPI_AINT ((MPI_Datatype)1275069499)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ANY_SOURCE">MPI_ANY_SOURCE	&lt;C: #define MPI_ANY_SOURCE (-2)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ANY_TAG">MPI_ANY_TAG	&lt;C: #define MPI_ANY_TAG (-1)	&gt;</body>

<body package="StMPI GUI" selector="MPI_APPNUM">MPI_APPNUM	&lt;C: #define MPI_APPNUM 1681915917	&gt;</body>

<body package="StMPI GUI" selector="MPI_ARGVS_NULL">MPI_ARGVS_NULL	&lt;C: #define MPI_ARGVS_NULL ((char***)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ARGV_NULL">MPI_ARGV_NULL	"&lt;C: #define MPI_ARGV_NULL ((char**)0)	&gt;"	^0</body>

<body package="StMPI GUI" selector="MPI_BAND">MPI_BAND	&lt;C: #define MPI_BAND ((MPI_Op)1476395014)	&gt;</body>

<body package="StMPI GUI" selector="MPI_BOR">MPI_BOR	&lt;C: #define MPI_BOR ((MPI_Op)1476395016)	&gt;</body>

<body package="StMPI GUI" selector="MPI_BOTTOM">MPI_BOTTOM	&lt;C: #define MPI_BOTTOM ((void*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_BSEND_OVERHEAD">MPI_BSEND_OVERHEAD	&lt;C: #define MPI_BSEND_OVERHEAD MSMPI_BSEND_OVERHEAD_V1	&gt;</body>

<body package="StMPI GUI" selector="MPI_BXOR">MPI_BXOR	&lt;C: #define MPI_BXOR ((MPI_Op)1476395018)	&gt;</body>

<body package="StMPI GUI" selector="MPI_BYTE">MPI_BYTE	&lt;C: #define MPI_BYTE ((MPI_Datatype)1275068685)	&gt;</body>

<body package="StMPI GUI" selector="MPI_CHAR">MPI_CHAR	&lt;C: #define MPI_CHAR ((MPI_Datatype)1275068673)	&gt;</body>

<body package="StMPI GUI" selector="MPI_CHARACTER">MPI_CHARACTER	&lt;C: #define MPI_CHARACTER ((MPI_Datatype)1275068698)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMM_DUP_FN">MPI_COMM_DUP_FN	&lt;C: #define MPI_COMM_DUP_FN ((MPI_Comm_copy_attr_function*)MPIR_Dup_fn)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMM_NULL">MPI_COMM_NULL	"&lt;C: #define MPI_COMM_NULL ((MPI_Comm)67108864)	&gt;"	^67108864</body>

<body package="StMPI GUI" selector="MPI_COMM_NULL_COPY_FN">MPI_COMM_NULL_COPY_FN	&lt;C: #define MPI_COMM_NULL_COPY_FN ((MPI_Comm_copy_attr_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMM_NULL_DELETE_FN">MPI_COMM_NULL_DELETE_FN	&lt;C: #define MPI_COMM_NULL_DELETE_FN ((MPI_Comm_delete_attr_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMM_SELF">MPI_COMM_SELF	"&lt;C: #define MPI_COMM_SELF ((MPI_Comm)1140850689)	&gt;"	^1140850689</body>

<body package="StMPI GUI" selector="MPI_COMM_WORLD">MPI_COMM_WORLD	&lt;C: #define MPI_COMM_WORLD ((MPI_Comm)1140850688)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMPLEX">MPI_COMPLEX	&lt;C: #define MPI_COMPLEX ((MPI_Datatype)1275070494)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMPLEX16">MPI_COMPLEX16	&lt;C: #define MPI_COMPLEX16 ((MPI_Datatype)1275072554)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMPLEX32">MPI_COMPLEX32	&lt;C: #define MPI_COMPLEX32 MPI_DATATYPE_NULL	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMPLEX4">MPI_COMPLEX4	&lt;C: #define MPI_COMPLEX4 MPI_DATATYPE_NULL	&gt;</body>

<body package="StMPI GUI" selector="MPI_COMPLEX8">MPI_COMPLEX8	&lt;C: #define MPI_COMPLEX8 ((MPI_Datatype)1275070504)	&gt;</body>

<body package="StMPI GUI" selector="MPI_CONGRUENT">MPI_CONGRUENT	&lt;C: #define MPI_CONGRUENT 1	&gt;</body>

<body package="StMPI GUI" selector="MPI_CONVERSION_FN_NULL">MPI_CONVERSION_FN_NULL	&lt;C: #define MPI_CONVERSION_FN_NULL ((MPI_Datarep_conversion_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_COUNT">MPI_COUNT	&lt;C: #define MPI_COUNT ((MPI_Datatype)1275070525)	&gt;</body>

<body package="StMPI GUI" selector="MPI_C_BOOL">MPI_C_BOOL	&lt;C: #define MPI_C_BOOL ((MPI_Datatype)1275068695)	&gt;</body>

<body package="StMPI GUI" selector="MPI_C_COMPLEX">MPI_C_COMPLEX	&lt;C: #define MPI_C_COMPLEX ((MPI_Datatype)1275070482)	&gt;</body>

<body package="StMPI GUI" selector="MPI_C_DOUBLE_COMPLEX">MPI_C_DOUBLE_COMPLEX	&lt;C: #define MPI_C_DOUBLE_COMPLEX ((MPI_Datatype)1275072532)	&gt;</body>

<body package="StMPI GUI" selector="MPI_C_FLOAT_COMPLEX">MPI_C_FLOAT_COMPLEX	&lt;C: #define MPI_C_FLOAT_COMPLEX ((MPI_Datatype)1275070483)	&gt;</body>

<body package="StMPI GUI" selector="MPI_C_LONG_DOUBLE_COMPLEX">MPI_C_LONG_DOUBLE_COMPLEX	&lt;C: #define MPI_C_LONG_DOUBLE_COMPLEX ((MPI_Datatype)1275072533)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DATATYPE_NULL">MPI_DATATYPE_NULL	&lt;C: #define MPI_DATATYPE_NULL ((MPI_Datatype)201326592)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DISPLACEMENT_CURRENT">MPI_DISPLACEMENT_CURRENT	&lt;C: #define MPI_DISPLACEMENT_CURRENT (-54278278)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DISTRIBUTE_BLOCK">MPI_DISTRIBUTE_BLOCK	&lt;C: #define MPI_DISTRIBUTE_BLOCK 121	&gt;</body>

<body package="StMPI GUI" selector="MPI_DISTRIBUTE_CYCLIC">MPI_DISTRIBUTE_CYCLIC	&lt;C: #define MPI_DISTRIBUTE_CYCLIC 122	&gt;</body>

<body package="StMPI GUI" selector="MPI_DISTRIBUTE_DFLT_DARG">MPI_DISTRIBUTE_DFLT_DARG	&lt;C: #define MPI_DISTRIBUTE_DFLT_DARG (-49767)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DISTRIBUTE_NONE">MPI_DISTRIBUTE_NONE	&lt;C: #define MPI_DISTRIBUTE_NONE 123	&gt;</body>

<body package="StMPI GUI" selector="MPI_DOUBLE">MPI_DOUBLE	&lt;C: #define MPI_DOUBLE ((MPI_Datatype)1275070475)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DOUBLE_COMPLEX">MPI_DOUBLE_COMPLEX	&lt;C: #define MPI_DOUBLE_COMPLEX ((MPI_Datatype)1275072546)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DOUBLE_INT">MPI_DOUBLE_INT	&lt;C: #define MPI_DOUBLE_INT ((MPI_Datatype)2348810241)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DOUBLE_PRECISION">MPI_DOUBLE_PRECISION	&lt;C: #define MPI_DOUBLE_PRECISION ((MPI_Datatype)1275070495)	&gt;</body>

<body package="StMPI GUI" selector="MPI_DUP_FN">MPI_DUP_FN	&lt;C: #define MPI_DUP_FN MPIR_Dup_fn	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERRCODES_IGNORE">MPI_ERRCODES_IGNORE	&lt;C: #define MPI_ERRCODES_IGNORE ((int*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERRHANDLER_NULL">MPI_ERRHANDLER_NULL	&lt;C: #define MPI_ERRHANDLER_NULL ((MPI_Errhandler)335544320)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERRORS_ARE_FATAL">MPI_ERRORS_ARE_FATAL	&lt;C: #define MPI_ERRORS_ARE_FATAL ((MPI_Errhandler)1409286144)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERRORS_RETURN">MPI_ERRORS_RETURN	&lt;C: #define MPI_ERRORS_RETURN ((MPI_Errhandler)1409286145)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_ACCESS">MPI_ERR_ACCESS	&lt;C: #define MPI_ERR_ACCESS 20	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_AMODE">MPI_ERR_AMODE	&lt;C: #define MPI_ERR_AMODE 21	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_ARG">MPI_ERR_ARG	&lt;C: #define MPI_ERR_ARG 12	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_ASSERT">MPI_ERR_ASSERT	&lt;C: #define MPI_ERR_ASSERT 53	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_BAD_FILE">MPI_ERR_BAD_FILE	&lt;C: #define MPI_ERR_BAD_FILE 22	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_BASE">MPI_ERR_BASE	&lt;C: #define MPI_ERR_BASE 46	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_BUFFER">MPI_ERR_BUFFER	&lt;C: #define MPI_ERR_BUFFER 1	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_COMM">MPI_ERR_COMM	&lt;C: #define MPI_ERR_COMM 5	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_CONVERSION">MPI_ERR_CONVERSION	&lt;C: #define MPI_ERR_CONVERSION 23	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_COUNT">MPI_ERR_COUNT	&lt;C: #define MPI_ERR_COUNT 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_DIMS">MPI_ERR_DIMS	&lt;C: #define MPI_ERR_DIMS 11	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_DISP">MPI_ERR_DISP	&lt;C: #define MPI_ERR_DISP 52	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_DUP_DATAREP">MPI_ERR_DUP_DATAREP	&lt;C: #define MPI_ERR_DUP_DATAREP 24	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_FILE">MPI_ERR_FILE	&lt;C: #define MPI_ERR_FILE 27	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_FILE_EXISTS">MPI_ERR_FILE_EXISTS	&lt;C: #define MPI_ERR_FILE_EXISTS 25	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_FILE_IN_USE">MPI_ERR_FILE_IN_USE	&lt;C: #define MPI_ERR_FILE_IN_USE 26	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_GROUP">MPI_ERR_GROUP	&lt;C: #define MPI_ERR_GROUP 8	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_INFO">MPI_ERR_INFO	&lt;C: #define MPI_ERR_INFO 28	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_INFO_KEY">MPI_ERR_INFO_KEY	&lt;C: #define MPI_ERR_INFO_KEY 29	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_INFO_NOKEY">MPI_ERR_INFO_NOKEY	&lt;C: #define MPI_ERR_INFO_NOKEY 31	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_INFO_VALUE">MPI_ERR_INFO_VALUE	&lt;C: #define MPI_ERR_INFO_VALUE 30	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_INTERN">MPI_ERR_INTERN	&lt;C: #define MPI_ERR_INTERN 16	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_IN_STATUS">MPI_ERR_IN_STATUS	&lt;C: #define MPI_ERR_IN_STATUS 17	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_IO">MPI_ERR_IO	&lt;C: #define MPI_ERR_IO 32	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_KEYVAL">MPI_ERR_KEYVAL	&lt;C: #define MPI_ERR_KEYVAL 48	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_LASTCODE">MPI_ERR_LASTCODE	&lt;C: #define MPI_ERR_LASTCODE 1073741823	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_LOCKTYPE">MPI_ERR_LOCKTYPE	&lt;C: #define MPI_ERR_LOCKTYPE 47	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_NAME">MPI_ERR_NAME	&lt;C: #define MPI_ERR_NAME 33	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_NOT_SAME">MPI_ERR_NOT_SAME	&lt;C: #define MPI_ERR_NOT_SAME 35	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_NO_MEM">MPI_ERR_NO_MEM	&lt;C: #define MPI_ERR_NO_MEM 34	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_NO_SPACE">MPI_ERR_NO_SPACE	&lt;C: #define MPI_ERR_NO_SPACE 36	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_NO_SUCH_FILE">MPI_ERR_NO_SUCH_FILE	&lt;C: #define MPI_ERR_NO_SUCH_FILE 37	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_OP">MPI_ERR_OP	&lt;C: #define MPI_ERR_OP 9	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_OTHER">MPI_ERR_OTHER	&lt;C: #define MPI_ERR_OTHER 15	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_PENDING">MPI_ERR_PENDING	&lt;C: #define MPI_ERR_PENDING 18	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_PORT">MPI_ERR_PORT	&lt;C: #define MPI_ERR_PORT 38	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_QUOTA">MPI_ERR_QUOTA	&lt;C: #define MPI_ERR_QUOTA 39	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_RANK">MPI_ERR_RANK	&lt;C: #define MPI_ERR_RANK 6	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_READ_ONLY">MPI_ERR_READ_ONLY	&lt;C: #define MPI_ERR_READ_ONLY 40	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_REQUEST">MPI_ERR_REQUEST	&lt;C: #define MPI_ERR_REQUEST 19	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_RMA_CONFLICT">MPI_ERR_RMA_CONFLICT	&lt;C: #define MPI_ERR_RMA_CONFLICT 49	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_RMA_SYNC">MPI_ERR_RMA_SYNC	&lt;C: #define MPI_ERR_RMA_SYNC 50	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_ROOT">MPI_ERR_ROOT	&lt;C: #define MPI_ERR_ROOT 7	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_SERVICE">MPI_ERR_SERVICE	&lt;C: #define MPI_ERR_SERVICE 41	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_SIZE">MPI_ERR_SIZE	&lt;C: #define MPI_ERR_SIZE 51	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_SPAWN">MPI_ERR_SPAWN	&lt;C: #define MPI_ERR_SPAWN 42	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_TAG">MPI_ERR_TAG	&lt;C: #define MPI_ERR_TAG 4	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_TOPOLOGY">MPI_ERR_TOPOLOGY	&lt;C: #define MPI_ERR_TOPOLOGY 10	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_TRUNCATE">MPI_ERR_TRUNCATE	&lt;C: #define MPI_ERR_TRUNCATE 14	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_TYPE">MPI_ERR_TYPE	&lt;C: #define MPI_ERR_TYPE 3	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_UNKNOWN">MPI_ERR_UNKNOWN	&lt;C: #define MPI_ERR_UNKNOWN 13	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_UNSUPPORTED_DATAREP">MPI_ERR_UNSUPPORTED_DATAREP	&lt;C: #define MPI_ERR_UNSUPPORTED_DATAREP 43	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_UNSUPPORTED_OPERATION">MPI_ERR_UNSUPPORTED_OPERATION	&lt;C: #define MPI_ERR_UNSUPPORTED_OPERATION 44	&gt;</body>

<body package="StMPI GUI" selector="MPI_ERR_WIN">MPI_ERR_WIN	&lt;C: #define MPI_ERR_WIN 45	&gt;</body>

<body package="StMPI GUI" selector="MPI_FILE_NULL">MPI_FILE_NULL	&lt;C: #define MPI_FILE_NULL ((MPI_File)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_FLOAT">MPI_FLOAT	&lt;C: #define MPI_FLOAT ((MPI_Datatype)1275069450)	&gt;</body>

<body package="StMPI GUI" selector="MPI_FLOAT_INT">MPI_FLOAT_INT	&lt;C: #define MPI_FLOAT_INT ((MPI_Datatype)2348810240)	&gt;</body>

<body package="StMPI GUI" selector="MPI_GROUP_EMPTY">MPI_GROUP_EMPTY	&lt;C: #define MPI_GROUP_EMPTY ((MPI_Group)1207959552)	&gt;</body>

<body package="StMPI GUI" selector="MPI_GROUP_NULL">MPI_GROUP_NULL	&lt;C: #define MPI_GROUP_NULL ((MPI_Group)134217728)	&gt;</body>

<body package="StMPI GUI" selector="MPI_HOST">MPI_HOST	&lt;C: #define MPI_HOST 1681915907	&gt;</body>

<body package="StMPI GUI" selector="MPI_IDENT">MPI_IDENT	&lt;C: #define MPI_IDENT 0	&gt;</body>

<body package="StMPI GUI" selector="MPI_INCLUDED">MPI_INCLUDED	&lt;C: #define MPI_INCLUDED 	&gt;</body>

<body package="StMPI GUI" selector="MPI_INFO_NULL">MPI_INFO_NULL	"&lt;C: #define MPI_INFO_NULL ((MPI_Info)469762048)	&gt;"	^469762048</body>

<body package="StMPI GUI" selector="MPI_INT">MPI_INT	&lt;C: #define MPI_INT ((MPI_Datatype)1275069445)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INT16_T">MPI_INT16_T	&lt;C: #define MPI_INT16_T ((MPI_Datatype)1275068980)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INT32_T">MPI_INT32_T	&lt;C: #define MPI_INT32_T ((MPI_Datatype)1275069493)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INT64_T">MPI_INT64_T	&lt;C: #define MPI_INT64_T ((MPI_Datatype)1275070518)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INT8_T">MPI_INT8_T	&lt;C: #define MPI_INT8_T ((MPI_Datatype)1275068723)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INTEGER">MPI_INTEGER	&lt;C: #define MPI_INTEGER ((MPI_Datatype)1275069467)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INTEGER1">MPI_INTEGER1	&lt;C: #define MPI_INTEGER1 ((MPI_Datatype)1275068717)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INTEGER16">MPI_INTEGER16	&lt;C: #define MPI_INTEGER16 MPI_DATATYPE_NULL	&gt;</body>

<body package="StMPI GUI" selector="MPI_INTEGER2">MPI_INTEGER2	&lt;C: #define MPI_INTEGER2 ((MPI_Datatype)1275068975)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INTEGER4">MPI_INTEGER4	&lt;C: #define MPI_INTEGER4 ((MPI_Datatype)1275069488)	&gt;</body>

<body package="StMPI GUI" selector="MPI_INTEGER8">MPI_INTEGER8	&lt;C: #define MPI_INTEGER8 ((MPI_Datatype)1275070513)	&gt;</body>

<body package="StMPI GUI" selector="MPI_IN_PLACE">MPI_IN_PLACE	&lt;C: #define MPI_IN_PLACE ((void*)(MPI_Aint)-1)	&gt;</body>

<body package="StMPI GUI" selector="MPI_IO">MPI_IO	&lt;C: #define MPI_IO 1681915909	&gt;</body>

<body package="StMPI GUI" selector="MPI_KEYVAL_INVALID">MPI_KEYVAL_INVALID	&lt;C: #define MPI_KEYVAL_INVALID 603979776	&gt;</body>

<body package="StMPI GUI" selector="MPI_LAND">MPI_LAND	&lt;C: #define MPI_LAND ((MPI_Op)1476395013)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LASTUSEDCODE">MPI_LASTUSEDCODE	&lt;C: #define MPI_LASTUSEDCODE 1681915915	&gt;</body>

<body package="StMPI GUI" selector="MPI_LB">MPI_LB	&lt;C: #define MPI_LB ((MPI_Datatype)1275068432)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LOCK_EXCLUSIVE">MPI_LOCK_EXCLUSIVE	&lt;C: #define MPI_LOCK_EXCLUSIVE 234	&gt;</body>

<body package="StMPI GUI" selector="MPI_LOCK_SHARED">MPI_LOCK_SHARED	&lt;C: #define MPI_LOCK_SHARED 235	&gt;</body>

<body package="StMPI GUI" selector="MPI_LOGICAL">MPI_LOGICAL	&lt;C: #define MPI_LOGICAL ((MPI_Datatype)1275069469)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LONG">MPI_LONG	"&lt;C: #define MPI_LONG ((MPI_Datatype)1275069447)	&gt;"		^1275069447</body>

<body package="StMPI GUI" selector="MPI_LONG_DOUBLE">MPI_LONG_DOUBLE	&lt;C: #define MPI_LONG_DOUBLE ((MPI_Datatype)1275070476)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LONG_DOUBLE_INT">MPI_LONG_DOUBLE_INT	&lt;C: #define MPI_LONG_DOUBLE_INT ((MPI_Datatype)2348810244)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LONG_INT">MPI_LONG_INT	&lt;C: #define MPI_LONG_INT ((MPI_Datatype)2348810242)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LONG_LONG">MPI_LONG_LONG	&lt;C: #define MPI_LONG_LONG MPI_LONG_LONG_INT	&gt;</body>

<body package="StMPI GUI" selector="MPI_LONG_LONG_INT">MPI_LONG_LONG_INT	&lt;C: #define MPI_LONG_LONG_INT ((MPI_Datatype)1275070473)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LOR">MPI_LOR	&lt;C: #define MPI_LOR ((MPI_Op)1476395015)	&gt;</body>

<body package="StMPI GUI" selector="MPI_LXOR">MPI_LXOR	&lt;C: #define MPI_LXOR ((MPI_Op)1476395017)	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX">MPI_MAX	&lt;C: #define MPI_MAX ((MPI_Op)1476395009)	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAXLOC">MPI_MAXLOC	&lt;C: #define MPI_MAXLOC ((MPI_Op)1476395020)	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_DATAREP_STRING">MPI_MAX_DATAREP_STRING	&lt;C: #define MPI_MAX_DATAREP_STRING 128	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_ERROR_STRING">MPI_MAX_ERROR_STRING	&lt;C: #define MPI_MAX_ERROR_STRING 512	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_INFO_KEY">MPI_MAX_INFO_KEY	&lt;C: #define MPI_MAX_INFO_KEY 255	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_INFO_VAL">MPI_MAX_INFO_VAL	&lt;C: #define MPI_MAX_INFO_VAL 1024	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_LIBRARY_VERSION_STRING">MPI_MAX_LIBRARY_VERSION_STRING	&lt;C: #define MPI_MAX_LIBRARY_VERSION_STRING 64	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_OBJECT_NAME">MPI_MAX_OBJECT_NAME	&lt;C: #define MPI_MAX_OBJECT_NAME 128	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_PORT_NAME">MPI_MAX_PORT_NAME	&lt;C: #define MPI_MAX_PORT_NAME 256	&gt;</body>

<body package="StMPI GUI" selector="MPI_MAX_PROCESSOR_NAME">MPI_MAX_PROCESSOR_NAME	&lt;C: #define MPI_MAX_PROCESSOR_NAME 128	&gt;</body>

<body package="StMPI GUI" selector="MPI_MESSAGE_NO_PROC">MPI_MESSAGE_NO_PROC	&lt;C: #define MPI_MESSAGE_NO_PROC ((MPI_Message)1879048192)	&gt;</body>

<body package="StMPI GUI" selector="MPI_MESSAGE_NULL">MPI_MESSAGE_NULL	&lt;C: #define MPI_MESSAGE_NULL ((MPI_Message)805306368)	&gt;</body>

<body package="StMPI GUI" selector="MPI_METHOD">MPI_METHOD	&lt;C: #define MPI_METHOD _Success_(return==MPI_SUCCESS)int MPIAPI	&gt;</body>

<body package="StMPI GUI" selector="MPI_MIN">MPI_MIN	&lt;C: #define MPI_MIN ((MPI_Op)1476395010)	&gt;</body>

<body package="StMPI GUI" selector="MPI_MINLOC">MPI_MINLOC	&lt;C: #define MPI_MINLOC ((MPI_Op)1476395019)	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_APPEND">MPI_MODE_APPEND	&lt;C: #define MPI_MODE_APPEND 0x00000080	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_CREATE">MPI_MODE_CREATE	&lt;C: #define MPI_MODE_CREATE 0x00000001	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_DELETE_ON_CLOSE">MPI_MODE_DELETE_ON_CLOSE	&lt;C: #define MPI_MODE_DELETE_ON_CLOSE 0x00000010	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_EXCL">MPI_MODE_EXCL	&lt;C: #define MPI_MODE_EXCL 0x00000040	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_NOCHECK">MPI_MODE_NOCHECK	&lt;C: #define MPI_MODE_NOCHECK 1024	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_NOPRECEDE">MPI_MODE_NOPRECEDE	&lt;C: #define MPI_MODE_NOPRECEDE 8192	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_NOPUT">MPI_MODE_NOPUT	&lt;C: #define MPI_MODE_NOPUT 4096	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_NOSTORE">MPI_MODE_NOSTORE	&lt;C: #define MPI_MODE_NOSTORE 2048	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_NOSUCCEED">MPI_MODE_NOSUCCEED	&lt;C: #define MPI_MODE_NOSUCCEED 16384	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_RDONLY">MPI_MODE_RDONLY	&lt;C: #define MPI_MODE_RDONLY 0x00000002	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_RDWR">MPI_MODE_RDWR	&lt;C: #define MPI_MODE_RDWR 0x00000008	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_SEQUENTIAL">MPI_MODE_SEQUENTIAL	&lt;C: #define MPI_MODE_SEQUENTIAL 0x00000100	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_UNIQUE_OPEN">MPI_MODE_UNIQUE_OPEN	&lt;C: #define MPI_MODE_UNIQUE_OPEN 0x00000020	&gt;</body>

<body package="StMPI GUI" selector="MPI_MODE_WRONLY">MPI_MODE_WRONLY	&lt;C: #define MPI_MODE_WRONLY 0x00000004	&gt;</body>

<body package="StMPI GUI" selector="MPI_NO_OP">MPI_NO_OP	&lt;C: #define MPI_NO_OP ((MPI_Op)1476395022)	&gt;</body>

<body package="StMPI GUI" selector="MPI_NULL_COPY_FN">MPI_NULL_COPY_FN	&lt;C: #define MPI_NULL_COPY_FN ((MPI_Copy_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_NULL_DELETE_FN">MPI_NULL_DELETE_FN	&lt;C: #define MPI_NULL_DELETE_FN ((MPI_Delete_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_OFFSET">MPI_OFFSET	&lt;C: #define MPI_OFFSET ((MPI_Datatype)1275070524)	&gt;</body>

<body package="StMPI GUI" selector="MPI_OP_NULL">MPI_OP_NULL	&lt;C: #define MPI_OP_NULL ((MPI_Op)402653184)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ORDER_C">MPI_ORDER_C	&lt;C: #define MPI_ORDER_C 56	&gt;</body>

<body package="StMPI GUI" selector="MPI_ORDER_FORTRAN">MPI_ORDER_FORTRAN	&lt;C: #define MPI_ORDER_FORTRAN 57	&gt;</body>

<body package="StMPI GUI" selector="MPI_PACKED">MPI_PACKED	&lt;C: #define MPI_PACKED ((MPI_Datatype)1275068687)	&gt;</body>

<body package="StMPI GUI" selector="MPI_PROC_NULL">MPI_PROC_NULL	&lt;C: #define MPI_PROC_NULL (-1)	&gt;</body>

<body package="StMPI GUI" selector="MPI_PROD">MPI_PROD	&lt;C: #define MPI_PROD ((MPI_Op)1476395012)	&gt;</body>

<body package="StMPI GUI" selector="MPI_REAL">MPI_REAL	&lt;C: #define MPI_REAL ((MPI_Datatype)1275069468)	&gt;</body>

<body package="StMPI GUI" selector="MPI_REAL16">MPI_REAL16	&lt;C: #define MPI_REAL16 MPI_DATATYPE_NULL	&gt;</body>

<body package="StMPI GUI" selector="MPI_REAL2">MPI_REAL2	&lt;C: #define MPI_REAL2 MPI_DATATYPE_NULL	&gt;</body>

<body package="StMPI GUI" selector="MPI_REAL4">MPI_REAL4	&lt;C: #define MPI_REAL4 ((MPI_Datatype)1275069479)	&gt;</body>

<body package="StMPI GUI" selector="MPI_REAL8">MPI_REAL8	&lt;C: #define MPI_REAL8 ((MPI_Datatype)1275070505)	&gt;</body>

<body package="StMPI GUI" selector="MPI_REPLACE">MPI_REPLACE	&lt;C: #define MPI_REPLACE ((MPI_Op)1476395021)	&gt;</body>

<body package="StMPI GUI" selector="MPI_REQUEST_NULL">MPI_REQUEST_NULL	&lt;C: #define MPI_REQUEST_NULL ((MPI_Request)738197504)	&gt;</body>

<body package="StMPI GUI" selector="MPI_ROOT">MPI_ROOT	&lt;C: #define MPI_ROOT (-3)	&gt;</body>

<body package="StMPI GUI" selector="MPI_SEEK_CUR">MPI_SEEK_CUR	&lt;C: #define MPI_SEEK_CUR 602	&gt;</body>

<body package="StMPI GUI" selector="MPI_SEEK_END">MPI_SEEK_END	&lt;C: #define MPI_SEEK_END 604	&gt;</body>

<body package="StMPI GUI" selector="MPI_SEEK_SET">MPI_SEEK_SET	&lt;C: #define MPI_SEEK_SET 600	&gt;</body>

<body package="StMPI GUI" selector="MPI_SHORT">MPI_SHORT	&lt;C: #define MPI_SHORT ((MPI_Datatype)1275068931)	&gt;</body>

<body package="StMPI GUI" selector="MPI_SHORT_INT">MPI_SHORT_INT	&lt;C: #define MPI_SHORT_INT ((MPI_Datatype)2348810243)	&gt;</body>

<body package="StMPI GUI" selector="MPI_SIGNED_CHAR">MPI_SIGNED_CHAR	&lt;C: #define MPI_SIGNED_CHAR ((MPI_Datatype)1275068696)	&gt;</body>

<body package="StMPI GUI" selector="MPI_SIMILAR">MPI_SIMILAR	&lt;C: #define MPI_SIMILAR 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_STATUSES_IGNORE">MPI_STATUSES_IGNORE	&lt;C: #define MPI_STATUSES_IGNORE ((MPI_Status*)(MPI_Aint)1)	&gt;</body>

<body package="StMPI GUI" selector="MPI_STATUS_IGNORE">MPI_STATUS_IGNORE    "#define MPI_STATUS_IGNORE (MPI_Status *)1"    ^1</body>

<body package="StMPI GUI" selector="MPI_SUBVERSION">MPI_SUBVERSION	&lt;C: #define MPI_SUBVERSION 0	&gt;</body>

<body package="StMPI GUI" selector="MPI_SUCCESS">MPI_SUCCESS	&lt;C: #define MPI_SUCCESS 0	&gt;</body>

<body package="StMPI GUI" selector="MPI_SUM">MPI_SUM	&lt;C: #define MPI_SUM ((MPI_Op)1476395011)	&gt;</body>

<body package="StMPI GUI" selector="MPI_TAG_UB">MPI_TAG_UB	&lt;C: #define MPI_TAG_UB 1681915905	&gt;</body>

<body package="StMPI GUI" selector="MPI_THREAD_FUNNELED">MPI_THREAD_FUNNELED	&lt;C: #define MPI_THREAD_FUNNELED 1	&gt;</body>

<body package="StMPI GUI" selector="MPI_THREAD_MULTIPLE">MPI_THREAD_MULTIPLE	&lt;C: #define MPI_THREAD_MULTIPLE 3	&gt;</body>

<body package="StMPI GUI" selector="MPI_THREAD_SERIALIZED">MPI_THREAD_SERIALIZED	&lt;C: #define MPI_THREAD_SERIALIZED 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_THREAD_SINGLE">MPI_THREAD_SINGLE	&lt;C: #define MPI_THREAD_SINGLE 0	&gt;</body>

<body package="StMPI GUI" selector="MPI_TYPECLASS_COMPLEX">MPI_TYPECLASS_COMPLEX	&lt;C: #define MPI_TYPECLASS_COMPLEX 3	&gt;</body>

<body package="StMPI GUI" selector="MPI_TYPECLASS_INTEGER">MPI_TYPECLASS_INTEGER	&lt;C: #define MPI_TYPECLASS_INTEGER 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_TYPECLASS_REAL">MPI_TYPECLASS_REAL	&lt;C: #define MPI_TYPECLASS_REAL 1	&gt;</body>

<body package="StMPI GUI" selector="MPI_TYPE_DUP_FN">MPI_TYPE_DUP_FN	&lt;C: #define MPI_TYPE_DUP_FN ((MPI_Type_copy_attr_function*)MPIR_Dup_fn)	&gt;</body>

<body package="StMPI GUI" selector="MPI_TYPE_NULL_COPY_FN">MPI_TYPE_NULL_COPY_FN	&lt;C: #define MPI_TYPE_NULL_COPY_FN ((MPI_Type_copy_attr_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_TYPE_NULL_DELETE_FN">MPI_TYPE_NULL_DELETE_FN	&lt;C: #define MPI_TYPE_NULL_DELETE_FN ((MPI_Type_delete_attr_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UB">MPI_UB	&lt;C: #define MPI_UB ((MPI_Datatype)1275068433)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UINT16_T">MPI_UINT16_T	&lt;C: #define MPI_UINT16_T ((MPI_Datatype)1275068984)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UINT32_T">MPI_UINT32_T	&lt;C: #define MPI_UINT32_T ((MPI_Datatype)1275069497)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UINT64_T">MPI_UINT64_T	&lt;C: #define MPI_UINT64_T ((MPI_Datatype)1275070522)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UINT8_T">MPI_UINT8_T	&lt;C: #define MPI_UINT8_T ((MPI_Datatype)1275068727)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNDEFINED">MPI_UNDEFINED	&lt;C: #define MPI_UNDEFINED (-32766)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNEQUAL">MPI_UNEQUAL	&lt;C: #define MPI_UNEQUAL 3	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNIVERSE_SIZE">MPI_UNIVERSE_SIZE	&lt;C: #define MPI_UNIVERSE_SIZE 1681915913	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNSIGNED">MPI_UNSIGNED	&lt;C: #define MPI_UNSIGNED ((MPI_Datatype)1275069446)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNSIGNED_CHAR">MPI_UNSIGNED_CHAR	&lt;C: #define MPI_UNSIGNED_CHAR ((MPI_Datatype)1275068674)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNSIGNED_LONG">MPI_UNSIGNED_LONG	&lt;C: #define MPI_UNSIGNED_LONG ((MPI_Datatype)1275069448)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNSIGNED_LONG_LONG">MPI_UNSIGNED_LONG_LONG	&lt;C: #define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)1275070489)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNSIGNED_SHORT">MPI_UNSIGNED_SHORT	&lt;C: #define MPI_UNSIGNED_SHORT ((MPI_Datatype)1275068932)	&gt;</body>

<body package="StMPI GUI" selector="MPI_UNWEIGHTED">MPI_UNWEIGHTED	&lt;C: #define MPI_UNWEIGHTED ((int*)1)	&gt;</body>

<body package="StMPI GUI" selector="MPI_VERSION">MPI_VERSION	&lt;C: #define MPI_VERSION 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_WCHAR">MPI_WCHAR	&lt;C: #define MPI_WCHAR ((MPI_Datatype)1275068942)	&gt;</body>

<body package="StMPI GUI" selector="MPI_WEIGHTS_EMPTY">MPI_WEIGHTS_EMPTY	&lt;C: #define MPI_WEIGHTS_EMPTY ((int*)2)	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_BASE">MPI_WIN_BASE	&lt;C: #define MPI_WIN_BASE 1711276033	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_CREATE_FLAVOR">MPI_WIN_CREATE_FLAVOR	&lt;C: #define MPI_WIN_CREATE_FLAVOR 1711276039	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_DISP_UNIT">MPI_WIN_DISP_UNIT	&lt;C: #define MPI_WIN_DISP_UNIT 1711276037	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_DUP_FN">MPI_WIN_DUP_FN	&lt;C: #define MPI_WIN_DUP_FN ((MPI_Win_copy_attr_function*)MPIR_Dup_fn)	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_FLAVOR_ALLOCATE">MPI_WIN_FLAVOR_ALLOCATE	&lt;C: #define MPI_WIN_FLAVOR_ALLOCATE 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_FLAVOR_CREATE">MPI_WIN_FLAVOR_CREATE	&lt;C: #define MPI_WIN_FLAVOR_CREATE 1	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_FLAVOR_DYNAMIC">MPI_WIN_FLAVOR_DYNAMIC	&lt;C: #define MPI_WIN_FLAVOR_DYNAMIC 3	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_FLAVOR_SHARED">MPI_WIN_FLAVOR_SHARED	&lt;C: #define MPI_WIN_FLAVOR_SHARED 4	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_MODEL">MPI_WIN_MODEL	&lt;C: #define MPI_WIN_MODEL 1711276041	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_NULL">MPI_WIN_NULL	&lt;C: #define MPI_WIN_NULL ((MPI_Win)536870912)	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_NULL_COPY_FN">MPI_WIN_NULL_COPY_FN	&lt;C: #define MPI_WIN_NULL_COPY_FN ((MPI_Win_copy_attr_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_NULL_DELETE_FN">MPI_WIN_NULL_DELETE_FN	&lt;C: #define MPI_WIN_NULL_DELETE_FN ((MPI_Win_delete_attr_function*)0)	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_SEPARATE">MPI_WIN_SEPARATE	&lt;C: #define MPI_WIN_SEPARATE 1	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_SIZE">MPI_WIN_SIZE	&lt;C: #define MPI_WIN_SIZE 1711276035	&gt;</body>

<body package="StMPI GUI" selector="MPI_WIN_UNIFIED">MPI_WIN_UNIFIED	&lt;C: #define MPI_WIN_UNIFIED 2	&gt;</body>

<body package="StMPI GUI" selector="MPI_WTIME_IS_GLOBAL">MPI_WTIME_IS_GLOBAL	&lt;C: #define MPI_WTIME_IS_GLOBAL 1681915911	&gt;</body>

<body package="StMPI GUI" selector="MSMPI_BSEND_OVERHEAD_V1">MSMPI_BSEND_OVERHEAD_V1	&lt;C: #define MSMPI_BSEND_OVERHEAD_V1 95	&gt;</body>

<body package="StMPI GUI" selector="MSMPI_BSEND_OVERHEAD_V2">MSMPI_BSEND_OVERHEAD_V2	&lt;C: #define MSMPI_BSEND_OVERHEAD_V2 MSMPI_BSEND_OVERHEAD_V1	&gt;</body>

<body package="StMPI GUI" selector="MSMPI_MODE_HIDDEN">MSMPI_MODE_HIDDEN	&lt;C: #define MSMPI_MODE_HIDDEN 0x00000200	&gt;</body>

<body package="StMPI GUI" selector="MSMPI_VER">MSMPI_VER	&lt;C: #define MSMPI_VER 256	&gt;</body>

<body package="StMPI GUI" selector="_In_">_In_	&lt;C: #define _In_ 	&gt;</body>

<body package="StMPI GUI" selector="_In_opt_">_In_opt_	&lt;C: #define _In_opt_ 	&gt;</body>

<body package="StMPI GUI" selector="_In_z_">_In_z_	&lt;C: #define _In_z_ 	&gt;</body>

<body package="StMPI GUI" selector="_Inout_">_Inout_	&lt;C: #define _Inout_ 	&gt;</body>

<body package="StMPI GUI" selector="_Inout_opt_">_Inout_opt_	&lt;C: #define _Inout_opt_ 	&gt;</body>

<body package="StMPI GUI" selector="_Notref_">_Notref_	&lt;C: #define _Notref_ 	&gt;</body>

<body package="StMPI GUI" selector="_Out_">_Out_	&lt;C: #define _Out_ 	&gt;</body>

<body package="StMPI GUI" selector="_Out_opt_">_Out_opt_	&lt;C: #define _Out_opt_ 	&gt;</body>

<body package="StMPI GUI" selector="_Post_invalid_">_Post_invalid_	&lt;C: #define _Post_invalid_ 	&gt;</body>

<body package="StMPI GUI" selector="_Pre_opt_valid_">_Pre_opt_valid_	&lt;C: #define _Pre_opt_valid_ 	&gt;</body>

<body package="StMPI GUI" selector="_Pre_valid_">_Pre_valid_	&lt;C: #define _Pre_valid_ 	&gt;</body>

<body package="StMPI GUI" selector="__DATE__">__DATE__	&lt;C: #define __DATE__ "August 8, 2019"	&gt;</body>

<body package="StMPI GUI" selector="__FILE__">__FILE__	&lt;C: #define __FILE__ c:\Users\HP\Desktop\mpi.h	&gt;</body>

<body package="StMPI GUI" selector="__LINE__">__LINE__	&lt;C: #define __LINE__ 0	&gt;</body>

<body package="StMPI GUI" selector="__STDC__">__STDC__	&lt;C: #define __STDC__ 1	&gt;</body>

<body package="StMPI GUI" selector="__TIME__">__TIME__	&lt;C: #define __TIME__ "7:14:29 AM"	&gt;</body>

<body package="StMPI GUI" selector="_mpi_out_flag_">_mpi_out_flag_	&lt;C: #define _mpi_out_flag_ _Out_ _Deref_out_range_(0,1)	&gt;</body>
</methods>

<methods>
<class-id>StMPI</class-id> <category>macros</category>

<body package="StMPI GUI" selector="MPI_Comm_c2f:">MPI_Comm_c2f: comm	&lt;C: #define MPI_Comm_c2f(comm) (MPI_Fint)(comm )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Comm_f2c:">MPI_Comm_f2c: comm	&lt;C: #define MPI_Comm_f2c(comm) (MPI_Comm)(comm )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Errhandler_c2f:">MPI_Errhandler_c2f: errhandler	&lt;C: #define MPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Errhandler_f2c:">MPI_Errhandler_f2c: errhandler	&lt;C: #define MPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Group_c2f:">MPI_Group_c2f: group	&lt;C: #define MPI_Group_c2f(group) (MPI_Fint)(group )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Group_f2c:">MPI_Group_f2c: group	&lt;C: #define MPI_Group_f2c(group) (MPI_Group)(group )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Info_c2f:">MPI_Info_c2f: info	&lt;C: #define MPI_Info_c2f(info) (MPI_Fint)(info )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Info_f2c:">MPI_Info_f2c: info	&lt;C: #define MPI_Info_f2c(info) (MPI_Info)(info )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Message_c2f:">MPI_Message_c2f: msg	&lt;C: #define MPI_Message_c2f(msg) (MPI_Fint)(msg )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Message_f2c:">MPI_Message_f2c: msg	&lt;C: #define MPI_Message_f2c(msg) (MPI_Message)(msg )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Op_c2f:">MPI_Op_c2f: op	&lt;C: #define MPI_Op_c2f(op) (MPI_Fint)(op )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Op_f2c:">MPI_Op_f2c: op	&lt;C: #define MPI_Op_f2c(op) (MPI_Op)(op )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Request_c2f:">MPI_Request_c2f: request	&lt;C: #define MPI_Request_c2f(request) (MPI_Fint)(request )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Request_f2c:">MPI_Request_f2c: request	&lt;C: #define MPI_Request_f2c(request) (MPI_Request)(request )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Type_c2f:">MPI_Type_c2f: datatype	&lt;C: #define MPI_Type_c2f(datatype) (MPI_Fint)(datatype )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Type_f2c:">MPI_Type_f2c: datatype	&lt;C: #define MPI_Type_f2c(datatype) (MPI_Datatype)(datatype )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Win_c2f:">MPI_Win_c2f: win	&lt;C: #define MPI_Win_c2f(win) (MPI_Fint)(win )	&gt;</body>

<body package="StMPI GUI" selector="MPI_Win_f2c:">MPI_Win_f2c: win	&lt;C: #define MPI_Win_f2c(win) (MPI_Win)(win )	&gt;</body>

<body package="StMPI GUI" selector="MSMPI_DEPRECATE_20:">MSMPI_DEPRECATE_20: x	&lt;C: #define MSMPI_DEPRECATE_20(x) __declspec(deprecated("Deprecated in MPI 2.0, use '"#x "'.  To disable deprecation, define MSMPI_NO_DEPRECATE_20."))	&gt;</body>

<body package="StMPI GUI" selector="MSMPI_Get_version">MSMPI_Get_version	&lt;C: #define MSMPI_Get_version() (MSMPI_VER)	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Comm_c2f:">PMPI_Comm_c2f: comm	&lt;C: #define PMPI_Comm_c2f(comm) (MPI_Fint)(comm )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Comm_f2c:">PMPI_Comm_f2c: comm	&lt;C: #define PMPI_Comm_f2c(comm) (MPI_Comm)(comm )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Errhandler_c2f:">PMPI_Errhandler_c2f: errhandler	&lt;C: #define PMPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Errhandler_f2c:">PMPI_Errhandler_f2c: errhandler	&lt;C: #define PMPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Group_c2f:">PMPI_Group_c2f: group	&lt;C: #define PMPI_Group_c2f(group) (MPI_Fint)(group )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Group_f2c:">PMPI_Group_f2c: group	&lt;C: #define PMPI_Group_f2c(group) (MPI_Group)(group )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Info_c2f:">PMPI_Info_c2f: info	&lt;C: #define PMPI_Info_c2f(info) (MPI_Fint)(info )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Info_f2c:">PMPI_Info_f2c: info	&lt;C: #define PMPI_Info_f2c(info) (MPI_Info)(info )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Message_c2f:">PMPI_Message_c2f: msg	&lt;C: #define PMPI_Message_c2f(msg) (MPI_Fint)(msg )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Message_f2c:">PMPI_Message_f2c: msg	&lt;C: #define PMPI_Message_f2c(msg) (MPI_Message)(msg )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Op_c2f:">PMPI_Op_c2f: op	&lt;C: #define PMPI_Op_c2f(op) (MPI_Fint)(op )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Op_f2c:">PMPI_Op_f2c: op	&lt;C: #define PMPI_Op_f2c(op) (MPI_Op)(op )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Request_c2f:">PMPI_Request_c2f: request	&lt;C: #define PMPI_Request_c2f(request) (MPI_Fint)(request )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Request_f2c:">PMPI_Request_f2c: request	&lt;C: #define PMPI_Request_f2c(request) (MPI_Request)(request )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Type_c2f:">PMPI_Type_c2f: datatype	&lt;C: #define PMPI_Type_c2f(datatype) (MPI_Fint)(datatype )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Type_f2c:">PMPI_Type_f2c: datatype	&lt;C: #define PMPI_Type_f2c(datatype) (MPI_Datatype)(datatype )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Win_c2f:">PMPI_Win_c2f: win	&lt;C: #define PMPI_Win_c2f(win) (MPI_Fint)(win )	&gt;</body>

<body package="StMPI GUI" selector="PMPI_Win_f2c:">PMPI_Win_f2c: win	&lt;C: #define PMPI_Win_f2c(win) (MPI_Win)(win )	&gt;</body>

<body package="StMPI GUI" selector="_Deref_in_range_:with:">_Deref_in_range_: x with: y	&lt;C: #define _Deref_in_range_(x, y) 	&gt;</body>

<body package="StMPI GUI" selector="_Deref_out_range_:with:">_Deref_out_range_: x with: y	&lt;C: #define _Deref_out_range_(x, y) 	&gt;</body>

<body package="StMPI GUI" selector="_In_range_:with:">_In_range_: x with: y	&lt;C: #define _In_range_(x, y) 	&gt;</body>

<body package="StMPI GUI" selector="_In_reads_:">_In_reads_: x	&lt;C: #define _In_reads_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_In_reads_bytes_opt_:">_In_reads_bytes_opt_: x	&lt;C: #define _In_reads_bytes_opt_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_In_reads_opt_:">_In_reads_opt_: x	&lt;C: #define _In_reads_opt_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_In_reads_z_:">_In_reads_z_: x	&lt;C: #define _In_reads_z_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Inout_updates_opt_:">_Inout_updates_opt_: x	&lt;C: #define _Inout_updates_opt_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Out_writes_bytes_opt_:">_Out_writes_bytes_opt_: x	&lt;C: #define _Out_writes_bytes_opt_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Out_writes_opt_:">_Out_writes_opt_: x	&lt;C: #define _Out_writes_opt_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Out_writes_to_opt_:with:">_Out_writes_to_opt_: x with: y	&lt;C: #define _Out_writes_to_opt_(x, y) 	&gt;</body>

<body package="StMPI GUI" selector="_Out_writes_z_:">_Out_writes_z_: x	&lt;C: #define _Out_writes_z_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Post_equal_to_:">_Post_equal_to_: x	&lt;C: #define _Post_equal_to_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Post_satisfies_:">_Post_satisfies_: x	&lt;C: #define _Post_satisfies_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Pre_satisfies_:">_Pre_satisfies_: x	&lt;C: #define _Pre_satisfies_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_Success_:">_Success_: x	&lt;C: #define _Success_(x) 	&gt;</body>

<body package="StMPI GUI" selector="_When_:with:">_When_: x with: y	&lt;C: #define _When_(x, y) 	&gt;</body>

<body package="StMPI GUI" selector="_mpi_coll_rank_:">_mpi_coll_rank_: param	&lt;C: #define _mpi_coll_rank_(param) _In_ _Pre_satisfies_(param ==MPI_ROOT || param &gt;=MPI_PROC_NULL)	&gt;</body>

<body package="StMPI GUI" selector="_mpi_out_:with:">_mpi_out_: param with: sentinel	&lt;C: #define _mpi_out_(param, sentinel) _Out_ _Post_satisfies_(*param == sentinel ||*param &gt;=0)	&gt;</body>

<body package="StMPI GUI" selector="_mpi_out_range_:with:with:">_mpi_out_range_: param with: sentinel with: ub	&lt;C: #define _mpi_out_range_(param, sentinel, ub) _Out_ _Post_satisfies_(*param == sentinel ||(ub &gt;0&amp;&amp;*param &gt;=0&amp;&amp;*param &lt;= ub ))	&gt;</body>

<body package="StMPI GUI" selector="_mpi_position_:">_mpi_position_: ub	&lt;C: #define _mpi_position_(ub) _Inout_ _Deref_in_range_(0, ub )_Deref_out_range_(0, ub )	&gt;</body>

<body package="StMPI GUI" selector="_mpi_reads_:">_mpi_reads_: size	&lt;C: #define _mpi_reads_(size) _When_(size !=0,_In_reads_(size ))	&gt;</body>

<body package="StMPI GUI" selector="_mpi_reads_bytes_:">_mpi_reads_bytes_: size	&lt;C: #define _mpi_reads_bytes_(size) _When_(size !=0,_In_reads_bytes_(size ))	&gt;</body>

<body package="StMPI GUI" selector="_mpi_updates_:">_mpi_updates_: size	&lt;C: #define _mpi_updates_(size) _When_(size !=0,_Inout_updates_(size ))	&gt;</body>

<body package="StMPI GUI" selector="_mpi_writes_:">_mpi_writes_: size	&lt;C: #define _mpi_writes_(size) _When_(size !=0,_Out_writes_(size ))	&gt;</body>

<body package="StMPI GUI" selector="_mpi_writes_bytes_:">_mpi_writes_bytes_: size	&lt;C: #define _mpi_writes_bytes_(size) _When_(size !=0,_Out_writes_bytes_(size ))	&gt;</body>

<body package="StMPI GUI" selector="_mpi_writes_to_:with:">_mpi_writes_to_: size with: count	&lt;C: #define _mpi_writes_to_(size, count) _When_(size !=0,_Out_writes_to_(size , count ))	&gt;</body>
</methods>

</st-source>
