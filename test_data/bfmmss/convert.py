from lxml import etree as ET
import os
import glob

ET.FunctionNamespace("http://exslt.org/regular-expressions").prefix = 're'

target = os.path.join(
    os.path.dirname(__file__)
)

xsl_xml = ET.fromstring("""<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:re="http://exslt.org/regular-expressions"
    xmlns:tei="http://www.tei-c.org/ns/1.0"
    exclude-result-prefixes="xs"
    version="1.0">
    <xsl:output method="text">
    </xsl:output>
    <xsl:template match="tei:w">
        <xsl:value-of select="re:replace(re:replace(./text(), '%', 'gi', ''), '\[\w+\]', 'gi', '')"/><xsl:text> </xsl:text>
    </xsl:template>
    <xsl:template match="/">
        <xsl:apply-templates select=".//tei:w"/>
    </xsl:template>
    <xsl:template match="text()"/>
</xsl:stylesheet>""")

xsl = ET.XSLT(xsl_xml)


for file in glob.glob(os.path.join(target, "xml", "*.xml")):
    output = os.path.join(target, "txt", os.path.basename(file)+".txt")
    with open(file) as io_input:
        xml = ET.parse(io_input)

        with open(output, "w") as io_output:
            io_output.write(xsl.tostring(xsl(xml)))
