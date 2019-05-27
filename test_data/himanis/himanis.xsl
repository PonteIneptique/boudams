<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xpath-default-namespace="http://www.tei-c.org/ns/1.0"
    exclude-result-prefixes="xs"
    version="2.0">
    <xsl:output method="text"/>
    <xsl:strip-space elements="*"/>
    <xsl:template match="/">
        <xsl:result-document method="text">
            <xsl:apply-templates select=".//body"/>
        </xsl:result-document>
    </xsl:template>
    <xsl:template match="body">
        <xsl:apply-templates select="./p"/>
    </xsl:template>
    <xsl:template match="p">
        <xsl:apply-templates />
        <xsl:text>
</xsl:text>
    </xsl:template>
    <xsl:template match="w">
        <xsl:apply-templates /><xsl:text> </xsl:text>
    </xsl:template>
    <xsl:template match="expan">
        <xsl:apply-templates />
    </xsl:template>
    <xsl:template match="ex" />
    <xsl:template match="expan">
        <xsl:apply-templates />
    </xsl:template>
    <xsl:template match="abbr" >
        <xsl:apply-templates />
    </xsl:template>
    <xsl:template match="note"/>
</xsl:stylesheet>