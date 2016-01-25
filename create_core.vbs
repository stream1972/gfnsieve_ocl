if WScript.Arguments.Count <> 2 then
	WScript.Echo "Usage: ... inputfile.cl output.tmp"
	WScript.Quit(1)
end if

set fso = CreateObject("Scripting.FileSystemObject")

set input  = fso.OpenTextFile(WScript.Arguments(0), 1)  ' ForReading
set output = fso.OpenTextFile(WScript.Arguments(1), 2, True) ' ForWriting, create

do while not input.AtEndOfStream
	s = input.ReadLine()
	s = Replace(s, "\",  "\\")
	s = Replace(s, """", "\""")
	output.WriteLine """" & s & """ ""\n"""
loop

input.Close()
output.Close()
