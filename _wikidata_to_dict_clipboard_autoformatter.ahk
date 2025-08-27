; This is a script for copy-pasting wikidata entries into a Python dict. An entry formatted like
; Media Bias/Fact Check ID (P9852)
; becomes
; "Media Bias/Fact Check ID": "P9852",


^v::
{
   Send "{Blind}{Ctrl up}{Shift up}{Alt up}"
   Sleep 20
   msg := Trim(RegExReplace(A_Clipboard, "(([^\s]+ ?)+) \(([PQ]\d+)\)", "`"$1`": `"$3`",`r`n"))
   SendInput msg
   Send "{Blind}{Ctrl up}{Shift up}{Alt up}"
}
return