<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:tei="http://www.tei-c.org/ns/1.0"
    exclude-result-prefixes="xs"
    version="1.0">
    <xsl:output method="text"/>
    <xsl:template match="/">
        <xsl:apply-templates select="//tei:div[@type='edition']" />
    </xsl:template>
    <xsl:template match="tei:ex" />
    <xsl:template match="tei:head"/>
</xsl:stylesheet>