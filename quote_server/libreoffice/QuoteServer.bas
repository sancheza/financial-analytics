' QuoteServer.bas - the GETQUOTE() cell function.
'
' Installed into My Macros > Standard as the module QuoteServer by
' install_calc_integration.py. Calc finds cell functions only in the Standard
' library, so this module hands each call to calc_quote_client.py.
'
' Usage in a cell:
'   =GETQUOTE(A1)            price of the ticker or CUSIP in A1
'   =GETQUOTE(A1; "ytw")     a bond field: price, ytw, term
'   =GETQUOTE($A2; B$1)      field named in a header cell
Option Explicit

Private oClient As Object

Private Function ClientScript() As Object
    Dim factory As Object, provider As Object
    If IsNull(oClient) Then
        factory = CreateUnoService("com.sun.star.script.provider.MasterScriptProviderFactory")
        provider = factory.createScriptProvider("")
        oClient = provider.getScript("vnd.sun.star.script:calc_quote_client.py$getquote?language=Python&location=user")
    End If
    ClientScript = oClient
End Function

Function GETQUOTE(symbol As Variant, Optional field As Variant) As Variant
    Dim fieldName As String, script As Object
    fieldName = "price"
    If Not IsMissing(field) Then fieldName = CStr(field)
    On Error GoTo Failed
    script = ClientScript()
    GETQUOTE = script.invoke(Array(CStr(symbol), fieldName), Array(), Array())
    Exit Function
Failed:
    oClient = Null
    GETQUOTE = "#ERR: CLIENT"
End Function
