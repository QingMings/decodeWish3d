'use strict';
/** @type {!Array} */
var _0x34b6 = ["FXAAShader", "\n", "join", "varying vec2 vUv;", "void main() {", "vUv = uv;", "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );", "}", "uniform sampler2D tDiffuse;", "uniform vec2 resolution;", "#define FXAA_REDUCE_MIN   (1.0/128.0)", "#define FXAA_REDUCE_MUL   (1.0/8.0)", "#define FXAA_SPAN_MAX     8.0", "vec3 rgbNW = texture2D( tDiffuse, ( vUv - resolution )).xyz;", "vec3 rgbNE = texture2D( tDiffuse, ( vUv + vec2( resolution.x, -resolution.y ) )).xyz;", 
"vec3 rgbSW = texture2D( tDiffuse, ( vUv + vec2( -resolution.x, resolution.y ) )).xyz;", "vec3 rgbSE = texture2D( tDiffuse, ( vUv + resolution )).xyz;", "vec4 rgbaM  = texture2D( tDiffuse,  vUv );", "vec3 rgbM  = rgbaM.xyz;", "vec3 luma = vec3( 0.299, 0.587, 0.114 );", "float lumaNW = dot( rgbNW, luma );", "float lumaNE = dot( rgbNE, luma );", "float lumaSW = dot( rgbSW, luma );", "float lumaSE = dot( rgbSE, luma );", "float lumaM  = dot( rgbM,  luma );", "float lumaMin = min( lumaM, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );", 
"float lumaMax = max( lumaM, max( max( lumaNW, lumaNE) , max( lumaSW, lumaSE ) ) );", "vec2 dir;", "dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));", "dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));", "float dirReduce = max( ( lumaNW + lumaNE + lumaSW + lumaSE ) * ( 0.25 * FXAA_REDUCE_MUL ), FXAA_REDUCE_MIN );", "float rcpDirMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + dirReduce );", "dir = min( vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),", "max( vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),", "dir * rcpDirMin)) * resolution;", 
"vec4 rgbA = (1.0/2.0) * (", "texture2D(tDiffuse,  vUv + dir * (1.0/3.0 - 0.5)) +", "texture2D(tDiffuse,  vUv+ dir * (2.0/3.0 - 0.5)));", "vec4 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (", "texture2D(tDiffuse,  vUv + dir * (0.0/3.0 - 0.5)) +", "texture2D(tDiffuse,  vUv + dir * (3.0/3.0 - 0.5)));", "float lumaB = dot(rgbB, vec4(luma, 0.0));", "if ( ( lumaB < lumaMin ) || ( lumaB > lumaMax ) ) {", "gl_FragColor = rgbA;", "} else {", "gl_FragColor = rgbB;", "SMAAShader", "0.1", "varying vec4 vOffset[ 3 ];", 
"void SMAAEdgeDetectionVS( vec2 texcoord ) {", "vOffset[ 0 ] = texcoord.xyxy + resolution.xyxy * vec4( -1.0, 0.0, 0.0,  1.0 );", "vOffset[ 1 ] = texcoord.xyxy + resolution.xyxy * vec4(  1.0, 0.0, 0.0, -1.0 );", "vOffset[ 2 ] = texcoord.xyxy + resolution.xyxy * vec4( -2.0, 0.0, 0.0,  2.0 );", "SMAAEdgeDetectionVS( vUv );", "vec4 SMAAColorEdgeDetectionPS( vec2 texcoord, vec4 offset[3], sampler2D colorTex ) {", "vec2 threshold = vec2( SMAA_THRESHOLD, SMAA_THRESHOLD );", "vec4 delta;", "vec3 C = texture2D( colorTex, texcoord ).rgb;", 
"vec3 Cleft = texture2D( colorTex, offset[0].xy ).rgb;", "vec3 t = abs( C - Cleft );", "delta.x = max( max( t.r, t.g ), t.b );", "vec3 Ctop = texture2D( colorTex, offset[0].zw ).rgb;", "t = abs( C - Ctop );", "delta.y = max( max( t.r, t.g ), t.b );", "vec2 edges = step( threshold, delta.xy );", "if ( dot( edges, vec2( 1.0, 1.0 ) ) == 0.0 )", "discard;", "vec3 Cright = texture2D( colorTex, offset[1].xy ).rgb;", "t = abs( C - Cright );", "delta.z = max( max( t.r, t.g ), t.b );", "vec3 Cbottom  = texture2D( colorTex, offset[1].zw ).rgb;", 
"t = abs( C - Cbottom );", "delta.w = max( max( t.r, t.g ), t.b );", "float maxDelta = max( max( max( delta.x, delta.y ), delta.z ), delta.w );", "vec3 Cleftleft  = texture2D( colorTex, offset[2].xy ).rgb;", "t = abs( C - Cleftleft );", "vec3 Ctoptop = texture2D( colorTex, offset[2].zw ).rgb;", "t = abs( C - Ctoptop );", "maxDelta = max( max( maxDelta, delta.z ), delta.w );", "edges.xy *= step( 0.5 * maxDelta, delta.xy );", "return vec4( edges, 0.0, 0.0 );", "gl_FragColor = SMAAColorEdgeDetectionPS( vUv, vOffset, tDiffuse );", 
"8", "16", "( 1.0 / vec2( 160.0, 560.0 ) )", "( 1.0 / 7.0 )", "varying vec2 vPixcoord;", "void SMAABlendingWeightCalculationVS( vec2 texcoord ) {", "vPixcoord = texcoord / resolution;", "vOffset[ 0 ] = texcoord.xyxy + resolution.xyxy * vec4( -0.25, 0.125, 1.25, 0.125 );", "vOffset[ 1 ] = texcoord.xyxy + resolution.xyxy * vec4( -0.125, 0.25, -0.125, -1.25 );", "vOffset[ 2 ] = vec4( vOffset[ 0 ].xz, vOffset[ 1 ].yw ) + vec4( -2.0, 2.0, -2.0, 2.0 ) * resolution.xxyy * float( SMAA_MAX_SEARCH_STEPS );", 
"SMAABlendingWeightCalculationVS( vUv );", "#define SMAASampleLevelZeroOffset( tex, coord, offset ) texture2D( tex, coord + float( offset ) * resolution, 0.0 )", "uniform sampler2D tArea;", "uniform sampler2D tSearch;", "varying vec4 vOffset[3];", "vec2 round( vec2 x ) {", "return sign( x ) * floor( abs( x ) + 0.5 );", "float SMAASearchLength( sampler2D searchTex, vec2 e, float bias, float scale ) {", "e.r = bias + e.r * scale;", "return 255.0 * texture2D( searchTex, e, 0.0 ).r;", "float SMAASearchXLeft( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {", 
"vec2 e = vec2( 0.0, 1.0 );", "for ( int i = 0; i < SMAA_MAX_SEARCH_STEPS; i ++ ) {", "e = texture2D( edgesTex, texcoord, 0.0 ).rg;", "texcoord -= vec2( 2.0, 0.0 ) * resolution;", "if ( ! ( texcoord.x > end && e.g > 0.8281 && e.r == 0.0 ) ) break;", "texcoord.x += 0.25 * resolution.x;", "texcoord.x += resolution.x;", "texcoord.x += 2.0 * resolution.x;", "texcoord.x -= resolution.x * SMAASearchLength(searchTex, e, 0.0, 0.5);", "return texcoord.x;", "float SMAASearchXRight( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {", 
"texcoord += vec2( 2.0, 0.0 ) * resolution;", "if ( ! ( texcoord.x < end && e.g > 0.8281 && e.r == 0.0 ) ) break;", "texcoord.x -= 0.25 * resolution.x;", "texcoord.x -= resolution.x;", "texcoord.x -= 2.0 * resolution.x;", "texcoord.x += resolution.x * SMAASearchLength( searchTex, e, 0.5, 0.5 );", "float SMAASearchYUp( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {", "vec2 e = vec2( 1.0, 0.0 );", "texcoord += vec2( 0.0, 2.0 ) * resolution;", "if ( ! ( texcoord.y > end && e.r > 0.8281 && e.g == 0.0 ) ) break;", 
"texcoord.y -= 0.25 * resolution.y;", "texcoord.y -= resolution.y;", "texcoord.y -= 2.0 * resolution.y;", "texcoord.y += resolution.y * SMAASearchLength( searchTex, e.gr, 0.0, 0.5 );", "return texcoord.y;", "float SMAASearchYDown( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {", "texcoord -= vec2( 0.0, 2.0 ) * resolution;", "if ( ! ( texcoord.y < end && e.r > 0.8281 && e.g == 0.0 ) ) break;", "texcoord.y += 0.25 * resolution.y;", "texcoord.y += resolution.y;", "texcoord.y += 2.0 * resolution.y;", 
"texcoord.y -= resolution.y * SMAASearchLength( searchTex, e.gr, 0.5, 0.5 );", "vec2 SMAAArea( sampler2D areaTex, vec2 dist, float e1, float e2, float offset ) {", "vec2 texcoord = float( SMAA_AREATEX_MAX_DISTANCE ) * round( 4.0 * vec2( e1, e2 ) ) + dist;", "texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord + ( 0.5 * SMAA_AREATEX_PIXEL_SIZE );", "texcoord.y += SMAA_AREATEX_SUBTEX_SIZE * offset;", "return texture2D( areaTex, texcoord, 0.0 ).rg;", "vec4 SMAABlendingWeightCalculationPS( vec2 texcoord, vec2 pixcoord, vec4 offset[ 3 ], sampler2D edgesTex, sampler2D areaTex, sampler2D searchTex, ivec4 subsampleIndices ) {", 
"vec4 weights = vec4( 0.0, 0.0, 0.0, 0.0 );", "vec2 e = texture2D( edgesTex, texcoord ).rg;", "if ( e.g > 0.0 ) {", "vec2 d;", "vec2 coords;", "coords.x = SMAASearchXLeft( edgesTex, searchTex, offset[ 0 ].xy, offset[ 2 ].x );", "coords.y = offset[ 1 ].y;", "d.x = coords.x;", "float e1 = texture2D( edgesTex, coords, 0.0 ).r;", "coords.x = SMAASearchXRight( edgesTex, searchTex, offset[ 0 ].zw, offset[ 2 ].y );", "d.y = coords.x;", "d = d / resolution.x - pixcoord.x;", "vec2 sqrt_d = sqrt( abs( d ) );", 
"coords.y -= 1.0 * resolution.y;", "float e2 = SMAASampleLevelZeroOffset( edgesTex, coords, ivec2( 1, 0 ) ).r;", "weights.rg = SMAAArea( areaTex, sqrt_d, e1, e2, float( subsampleIndices.y ) );", "if ( e.r > 0.0 ) {", "coords.y = SMAASearchYUp( edgesTex, searchTex, offset[ 1 ].xy, offset[ 2 ].z );", "coords.x = offset[ 0 ].x;", "d.x = coords.y;", "float e1 = texture2D( edgesTex, coords, 0.0 ).g;", "coords.y = SMAASearchYDown( edgesTex, searchTex, offset[ 1 ].zw, offset[ 2 ].w );", "d.y = coords.y;", 
"d = d / resolution.y - pixcoord.y;", "float e2 = SMAASampleLevelZeroOffset( edgesTex, coords, ivec2( 0, 1 ) ).g;", "weights.ba = SMAAArea( areaTex, sqrt_d, e1, e2, float( subsampleIndices.x ) );", "return weights;", "gl_FragColor = SMAABlendingWeightCalculationPS( vUv, vPixcoord, vOffset, tDiffuse, tArea, tSearch, ivec4( 0.0 ) );", "varying vec4 vOffset[ 2 ];", "void SMAANeighborhoodBlendingVS( vec2 texcoord ) {", "vOffset[ 0 ] = texcoord.xyxy + resolution.xyxy * vec4( -1.0, 0.0, 0.0, 1.0 );", "vOffset[ 1 ] = texcoord.xyxy + resolution.xyxy * vec4( 1.0, 0.0, 0.0, -1.0 );", 
"SMAANeighborhoodBlendingVS( vUv );", "uniform sampler2D tColor;", "vec4 SMAANeighborhoodBlendingPS( vec2 texcoord, vec4 offset[ 2 ], sampler2D colorTex, sampler2D blendTex ) {", "vec4 a;", "a.xz = texture2D( blendTex, texcoord ).xz;", "a.y = texture2D( blendTex, offset[ 1 ].zw ).g;", "a.w = texture2D( blendTex, offset[ 1 ].xy ).a;", "if ( dot(a, vec4( 1.0, 1.0, 1.0, 1.0 )) < 1e-5 ) {", "return texture2D( colorTex, texcoord, 0.0 );", "vec2 offset;", "offset.x = a.a > a.b ? a.a : -a.b;", "offset.y = a.g > a.r ? -a.g : a.r;", 
"if ( abs( offset.x ) > abs( offset.y )) {", "offset.y = 0.0;", "offset.x = 0.0;", "vec4 C = texture2D( colorTex, texcoord, 0.0 );", "texcoord += sign( offset ) * resolution;", "vec4 Cop = texture2D( colorTex, texcoord, 0.0 );", "float s = abs( offset.x ) > abs( offset.y ) ? abs( offset.x ) : abs( offset.y );", "C.xyz = pow(C.xyz, vec3(2.2));", "Cop.xyz = pow(Cop.xyz, vec3(2.2));", "vec4 mixed = mix(C, Cop, s);", "mixed.xyz = pow(mixed.xyz, vec3(1.0 / 2.2));", "return mixed;", "gl_FragColor = SMAANeighborhoodBlendingPS( vUv, vOffset, tColor, tDiffuse );", 
"CopyShader", "uniform float opacity;", "vec4 texel = texture2D( tDiffuse, vUv );", "gl_FragColor = opacity * texel;", "EffectComposer", "renderer", "LinearFilter", "RGBAFormat", "getSize", "width", "height", "renderTarget1", "renderTarget2", "clone", "writeBuffer", "readBuffer", "passes", "THREE.EffectComposer relies on THREE.CopyShader", "error", "copyPass", "prototype", "push", "setSize", "splice", "length", "enabled", "render", "needsSwap", "context", "stencilFunc", "swapBuffers", "MaskPass", 
"ClearMaskPass", "dispose", "assign", "Pass", "clear", "renderToScreen", "THREE.Pass: .render() must be implemented in derived pass.", "call", "scene", "camera", "inverse", "create", "state", "setMask", "color", "buffers", "depth", "setLocked", "setTest", "stencil", "setOp", "setFunc", "setClear", "RenderPass", "overrideMaterial", "clearColor", "clearAlpha", "autoClear", "getHex", "getClearColor", "getClearAlpha", "setClearColor", "ShaderPass", "textureID", "tDiffuse", "ShaderMaterial", "uniforms", 
"material", "UniformsUtils", "defines", "vertexShader", "fragmentShader", "quad", "add", "value", "texture", "OutlinePass", "renderScene", "renderCamera", "selectedObjects", "visibleEdgeColor", "hiddenEdgeColor", "edgeGlow", "usePatternTexture", "edgeThickness", "edgeStrength", "downSampleRatio", "pulsePeriod", "resolution", "x", "y", "round", "maskBufferMaterial", "side", "DoubleSide", "renderTargetMaskBuffer", "generateMipmaps", "depthMaterial", "depthPacking", "RGBADepthPacking", "blending", "NoBlending", 
"prepareMaskMaterial", "getPrepareMaskMaterial", "renderTargetDepthBuffer", "renderTargetMaskDownSampleBuffer", "renderTargetBlurBuffer1", "renderTargetBlurBuffer2", "edgeDetectionMaterial", "getEdgeDetectionMaterial", "renderTargetEdgeBuffer1", "renderTargetEdgeBuffer2", "separableBlurMaterial1", "getSeperableBlurMaterial", "texSize", "kernelRadius", "separableBlurMaterial2", "overlayMaterial", "getOverlayMaterial", "THREE.OutlinePass relies on THREE.CopyShader", "copyUniforms", "opacity", "materialCopy", 
"oldClearColor", "oldClearAlpha", "tempPulseColor1", "tempPulseColor2", "textureMatrix", "Mesh", "visible", "traverse", "id", "bVisible", "set", "projectionMatrix", "multiply", "matrixWorldInverse", "copy", "disable", "changeVisibilityOfSelectedObjects", "updateTextureMatrix", "changeVisibilityOfNonSelectedObjects", "background", "cameraNearFar", "near", "far", "depthTexture", "now", "cos", "multiplyScalar", "maskTexture", "colorTexture", "direction", "BlurDirectionX", "BlurDirectionY", "edgeTexture1", 
"edgeTexture2", "patternTexture", "enable", "varying vec2 vUv;\n\t\t\t\tvarying vec4 projTexCoord;\n\t\t\t\tvarying vec4 vPosition;\n\t\t\t\tuniform mat4 textureMatrix;\n\t\t\t\tvoid main() {\n\t\t\t\t\tvUv = uv;\n\t\t\t\t\tvPosition = modelViewMatrix * vec4( position, 1.0 );\n\t\t\t\t\tvec4 worldPosition = modelMatrix * vec4( position, 1.0 );\n\t\t\t\t\tprojTexCoord = textureMatrix * worldPosition;\n\t\t\t\t\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n\t\t\t\t}", 
"#include <packing>\n\t\t\t\tvarying vec2 vUv;\n\t\t\t\tvarying vec4 vPosition;\n\t\t\t\tvarying vec4 projTexCoord;\n\t\t\t\tuniform sampler2D depthTexture;\n\t\t\t\tuniform vec2 cameraNearFar;\n\t\t\t\t\n\t\t\t\tvoid main() {\n\t\t\t\t\tfloat depth = unpackRGBAToDepth(texture2DProj( depthTexture, projTexCoord ));\n\t\t\t\t\tfloat viewZ = -perspectiveDepthToViewZ( depth, cameraNearFar.x, cameraNearFar.y );\n\t\t\t\t\tfloat depthTest = (-vPosition.z > viewZ) ? 1.0 : 0.0;\n\t\t\t\t\tgl_FragColor = vec4(0.0, depthTest, 1.0, 1.0);\n\t\t\t\t}", 
"varying vec2 vUv;\n\n\t\t\t\tvoid main() {\n\n\t\t\t\t\tvUv = uv;\n\n\t\t\t\t\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n\t\t\t\t}", "varying vec2 vUv;\n\t\t\t\tuniform sampler2D maskTexture;\n\t\t\t\tuniform vec2 texSize;\n\t\t\t\tuniform vec3 visibleEdgeColor;\n\t\t\t\tuniform vec3 hiddenEdgeColor;\n\t\t\t\t\n\t\t\t\tvoid main() {\n\n\t\t\t\t\tvec2 invSize = 1.0 / texSize;\n\t\t\t\t\tvec4 uvOffset = vec4(1.0, 0.0, 0.0, 1.0) * vec4(invSize, invSize);\n\t\t\t\t\tvec4 c1 = texture2D( maskTexture, vUv + uvOffset.xy);\n\t\t\t\t\tvec4 c2 = texture2D( maskTexture, vUv - uvOffset.xy);\n\t\t\t\t\tvec4 c3 = texture2D( maskTexture, vUv + uvOffset.yw);\n\t\t\t\t\tvec4 c4 = texture2D( maskTexture, vUv - uvOffset.yw);\n\t\t\t\t\tfloat diff1 = (c1.r - c2.r)*0.5;\n\t\t\t\t\tfloat diff2 = (c3.r - c4.r)*0.5;\n\t\t\t\t\tfloat d = length( vec2(diff1, diff2) );\n\t\t\t\t\tfloat a1 = min(c1.g, c2.g);\n\t\t\t\t\tfloat a2 = min(c3.g, c4.g);\n\t\t\t\t\tfloat visibilityFactor = min(a1, a2);\n\t\t\t\t\tvec3 edgeColor = 1.0 - visibilityFactor > 0.001 ? visibleEdgeColor : hiddenEdgeColor;\n\t\t\t\t\tgl_FragColor = vec4(edgeColor, 1.0) * vec4(d);\n\t\t\t\t}", 
"#include <common>\n\t\t\t\tvarying vec2 vUv;\n\t\t\t\tuniform sampler2D colorTexture;\n\t\t\t\tuniform vec2 texSize;\n\t\t\t\tuniform vec2 direction;\n\t\t\t\tuniform float kernelRadius;\n\t\t\t\t\n\t\t\t\tfloat gaussianPdf(in float x, in float sigma) {\n\t\t\t\t\treturn 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;\n\t\t\t\t}\n\t\t\t\tvoid main() {\n\t\t\t\t\tvec2 invSize = 1.0 / texSize;\n\t\t\t\t\tfloat weightSum = gaussianPdf(0.0, kernelRadius);\n\t\t\t\t\tvec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;\n\t\t\t\t\tvec2 delta = direction * invSize * kernelRadius/float(MAX_RADIUS);\n\t\t\t\t\tvec2 uvOffset = delta;\n\t\t\t\t\tfor( int i = 1; i <= MAX_RADIUS; i ++ ) {\n\t\t\t\t\t\tfloat w = gaussianPdf(uvOffset.x, kernelRadius);\n\t\t\t\t\t\tvec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;\n\t\t\t\t\t\tvec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;\n\t\t\t\t\t\tdiffuseSum += ((sample1 + sample2) * w);\n\t\t\t\t\t\tweightSum += (2.0 * w);\n\t\t\t\t\t\tuvOffset += delta;\n\t\t\t\t\t}\n\t\t\t\t\tgl_FragColor = vec4(diffuseSum/weightSum, 1.0);\n\t\t\t\t}", 
"varying vec2 vUv;\n\t\t\t\tuniform sampler2D maskTexture;\n\t\t\t\tuniform sampler2D edgeTexture1;\n\t\t\t\tuniform sampler2D edgeTexture2;\n\t\t\t\tuniform sampler2D patternTexture;\n\t\t\t\tuniform float edgeStrength;\n\t\t\t\tuniform float edgeGlow;\n\t\t\t\tuniform bool usePatternTexture;\n\t\t\t\t\n\t\t\t\tvoid main() {\n\t\t\t\t\tvec4 edgeValue1 = texture2D(edgeTexture1, vUv);\n\t\t\t\t\tvec4 edgeValue2 = texture2D(edgeTexture2, vUv);\n\t\t\t\t\tvec4 maskColor = texture2D(maskTexture, vUv);\n\t\t\t\t\tvec4 patternColor = texture2D(patternTexture, 6.0 * vUv);\n\t\t\t\t\tfloat visibilityFactor = 1.0 - maskColor.g > 0.0 ? 1.0 : 0.5;\n\t\t\t\t\tvec4 edgeValue = edgeValue1 + edgeValue2 * edgeGlow;\n\t\t\t\t\tvec4 finalColor = edgeStrength * edgeValue;\n\t\t\t\t\tif(usePatternTexture)\n\t\t\t\t\t\tfinalColor += + visibilityFactor * (1.0 - maskColor.r) * (1.0 - patternColor.r);\n\t\t\t\t\tgl_FragColor = finalColor;\n\t\t\t\t}", 
"AdditiveBlending", "undefined", "object", "exports", "global", "window", "-", "Overflow: input needs wider integers to process", "Illegal input >= 0x80 (not a basic code point)", "Invalid input", "floor", "fromCharCode", ".", "split", "charCodeAt", "", "lastIndexOf", "not-basic", "invalid-input", "overflow", "test", "toLowerCase", "slice", "xn--", "1.2.4", "function", "amd", "punycode", "nodeType", "hasOwnProperty", "./log", "defaultView", "pageXOffset", "pageYOffset", "scrollTo", "getImageData", 
"2d", "getContext", "putImageData", "Unable to copy canvas content from", "nodeValue", "createTextNode", "cloneNode", "firstChild", "nodeName", "SCRIPT", "appendChild", "nextSibling", "_scrollTop", "scrollTop", "_scrollLeft", "scrollLeft", "CANVAS", "TEXTAREA", "SELECT", "documentElement", "javascriptEnabled", "iframe", "createElement", "className", "html2canvas-container", "visibility", "style", "hidden", "position", "fixed", "left", "-10000px", "top", "0px", "border", "0", "scrolling", "no", "body", 
"document", "contentWindow", "onload", "childNodes", "type", "view", "userAgent", "scrollY", "scrollX", "px", "absolute", "open", "<!DOCTYPE html><html></html>", "write", "adoptNode", "replaceChild", "close", "r", "g", "b", "a", "fromArray", "namedColor", "rgb", "rgba", "hex6", "hex3", "darken", "isTransparent", "isBlack", "isArray", "min", "match", "substring", "toString", "rgba(", ",", ")", "rgb(", "transparent", "isColor", "./support", "./renderers/canvas", "./imageloader", "./nodeparser", "./nodecontainer", 
"./utils", "./clone", "loadUrlDocument", "./proxy", "getBounds", "data-html2canvas-node", "logging", "options", "start", "async", "allowTaint", "removeContainer", "imageTimeout", "strict", "string", "proxy", "Proxy must be used when rendering url", "reject", "innerWidth", "innerHeight", "then", "setAttribute", "onrendered", "options.onrendered is deprecated, html2canvas returns a Promise containing the canvas", "ownerDocument", "CanvasRenderer", "NodeContainer", "log", "utils", "canvas", "No canvas support", 
"html2canvas", "Document cloned", "[", "='", "']", "removeAttribute", "querySelector", "onclone", "resolve", "Finished rendering", "ready", "removeChild", "parentNode", "Cleaned up container", "max", "Cropping canvas at:", "left:", "top:", "width:", "height:", "Resulting crop with width", "and height", "with x", "and y", "drawImage", "scrollWidth", "offsetWidth", "clientWidth", "scrollHeight", "offsetHeight", "clientHeight", "href", "smallImage", "src", "DummyImageContainer for", "promise", "image", 
"Initiating DummyImageContainer", "onerror", "complete", "div", "img", "span", "Hidden Text", "fontFamily", "fontSize", "margin", "padding", "verticalAlign", "baseline", "offsetTop", "lineHeight", "normal", "super", "lineWidth", "middle", "./font", "data", "getMetrics", "./core", "proxyLoad", "URL", "about:blank", "colorStops", "x0", "y0", "x1", "y1", "TYPES", "REGEXP_COLORSTOP", "tainted", "crossOrigin", "anonymous", "./imagecontainer", "./dummyimagecontainer", "./proxyimagecontainer", "./framecontainer", 
"./svgcontainer", "./svgnodecontainer", "./lineargradientcontainer", "./webkitgradientcontainer", "bind", "link", "support", "origin", "location", "getOrigin", "findImages", "loadImage", "addImage", "forEach", "node", "url", "concat", "IMG", "svg", "IFRAME", "reduce", "findBackgroundImage", "hasImageBackground", "filter", "parseBackgroundImages", "imageExists", "Added image #", "args", "method", "none", "isSVG", "replace", "isSameOrigin", "cors", "useCORS", "linear-gradient", "gradient", "isInline", 
"some", "protocol", "hostname", "port", "getPromise", "catch", "timeout", "get", "images", "fetch", "Succesfully loaded image #", "Failed loading image #", "map", "all", "Finished searching images", "Timed out loading image", "race", "./gradientcontainer", "./color", "apply", "LINEAR", "REGEXP_DIRECTION", "right", "bottom", "to", "center", "reverse", "%", "stop", "console", "ms", "html2canvas:", "parseBackgrounds", "offsetBounds", "parent", "stack", "bounds", "borders", "clip", "backgroundClip", 
"computedStyles", "colors", "styles", "backgroundImages", "transformData", "transformMatrix", "isPseudoElement", "cloneTo", "getOpacity", "cssFloat", "assignStack", "children", "isElementVisible", "TEXT_NODE", "display", "css", "data-html2canvas-ignore", "hasAttribute", "INPUT", "getAttribute", "before", ":before", ":after", "computedStyle", "prefixedCss", "webkit", "moz", "o", "toUpperCase", "substr", "getComputedStyle", "cssInt", "fontWeight", "bold", "parseClip", "backgroundImage", "cssList", 
"auto", " ", "trim", "parseBackgroundSize", "backgroundSize", "contain", "parseBackgroundPosition", "backgroundPosition", "parseBackgroundRepeat", "backgroundRepeat", "parseTextShadows", "textShadow", "parseTransform", "hasTransform", "parseBounds", "transformOrigin", "parseTransformMatrix", "transform", "1,0,0,1,0,0", "getValue", "tagName", "password", "\u2022", "placeholder", "MATRIX_PROPERTY", "TEXT_SHADOW_PROPERTY", "TEXT_SHADOW_VALUES", "CLIP", "selectedIndex", "text", "matrix", "matrix3d", 
"indexOf", "./textcontainer", "./pseudoelementcontainer", "./fontmetrics", "./stackingcontext", "Starting NodeParser", "range", "renderQueue", "rectangle", "backgroundColor", "visibile", "createPseudoHideStyles", "disableAnimations", "nodes", "getPseudoElements", "getChildren", "fontMetrics", "Fetched nodes, total:", "Calculate overflow clips", "calculateOverflowClips", "Start fetching images", "Images loaded, starting parsing", "Creating stacking contexts", "createStackingContexts", "Sorting stacking contexts", 
"sortStackingContexts", "parse", "Render queue created with ", " items", "paint", "renderIndex", "asyncRenderer", "appendToDOM", "parseBorders", "rect", "cleanDOM", "PSEUDO_HIDE_ELEMENT_CLASS_BEFORE", ':before { content: "" !important; display: none !important; }', "PSEUDO_HIDE_ELEMENT_CLASS_AFTER", ':after { content: "" !important; display: none !important; }', "createStyles", "* { -webkit-animation: none !important; -moz-animation: none !important; -o-animation: none !important; animation: none !important; ", 
"-webkit-transition: none !important; -moz-transition: none !important; -o-transition: none !important; transition: none !important;}", "innerHTML", "ELEMENT_NODE", "getPseudoElement", "content", "-moz-alt-content", "html2canvaspseudoelement", "item", "newStackingContext", "getParentStack", "contexts", "isRootElement", "isBodyWithTransparentRoot", "BODY", "sort", "parseTextBounds", "textDecoration", "rangeBounds", "getRangeBounds", "splitText", "getWrapperBounds", "html2canvaswrapper", "createRange", 
"setStart", "setEnd", "getBoundingClientRect", "restore", "ctx", "paintText", "paintNode", "setOpacity", "save", "setTransform", "checkbox", "paintCheckbox", "radio", "paintRadio", "paintElement", "renderBackground", "renderBorders", "renderImage", "Error loading <", ">", "Error loading <img>", "paintFormValue", "#A5A5A5", "#DEDEDE", "checked", "#424242", "arial", "font", "\u2714", "circleStroke", "ceil", "circle", "textAlign", "paddingLeft", "paddingTop", "paddingRight", "paddingBottom", "borderLeftStyle", 
"borderTopStyle", "borderLeftWidth", "borderTopWidth", "boxSizing", "whiteSpace", "wordWrap", "html2canvas: Parse: Exception caught in renderFormValue: ", "message", "textContent", "applyTextTransform", "decode", "ucs2", "letterRendering", "encode", "fontStyle", "fontVariant", "offsetX", "offsetY", "blur", "fontShadow", "clearShadow", "renderTextDecoration", "underline", "overline", "line-through", "Style", "Color", "inset", "Width", "Top", "Right", "Bottom", "Left", "parseBackgroundClip", "topLeftOuter", 
"topLeftInner", "topRightOuter", "topRightInner", "bottomRightOuter", "bottomRightInner", "bottomLeftOuter", "bottomLeftInner", "content-box", "padding-box", "sqrt", "subdivide", "topLeft", "topRight", "bottomRight", "bottomLeft", "bezierCurve", "line", "curveTo", "c1", "end", "curveToReversed", "c2", "c3", "c4", "zIndex", "inline", "inline-block", "inline-table", "letterSpacing", "Radius", "TopLeft", "TopRight", "BottomRight", "BottomLeft", "relative", "static", "float", "HEAD", "TITLE", "OBJECT", 
"BR", "OPTION", "./xhr", "decode64", "withCredentials", "No proxy configured", "data:", ";base64,", "script", "html2canvas_", "_", "random", "?url=", "&callback=html2canvas.proxy.", "text/html", "parseFromString", "DOMParser not supported, falling back to createHTMLDocument", "createHTMLDocument", "implementation", "createHTMLDocument write not supported, falling back to document.body.innerHTML", "base", "host", "head", "insertBefore", "Proxy", "ProxyURL", "Anonymous", "getHideClass", "PSEUDO_HIDE_ELEMENT_CLASS_", 
"BEFORE", "AFTER", "___html2canvas___pseudoelement_before", "___html2canvas___pseudoelement_after", "renderBackgroundColor", "renderBackgroundImage", "renderBorder", "drawShape", "renderBackgroundRepeating", "Error loading background-image", "renderBackgroundGradient", "Unknown background-image type", "repeat-x", "backgroundRepeatShape", "repeat no-repeat", "repeat-y", "no-repeat repeat", "no-repeat", "renderBackgroundRepeat", "../renderer", "../lineargradientcontainer", "../log", "taintCtx", "textBaseline", 
"variables", "Initialized CanvasRenderer with size", "setFillStyle", "fillStyle", "fillRect", "beginPath", "PI", "arc", "closePath", "fill", "strokeStyle", "stroke", "shape", "taints", "moveTo", "To", "shadowBlur", "setVariable", "shadowOffsetX", "shadowOffsetY", "shadowColor", "rgba(0,0,0,0)", "globalAlpha", "translate", "fillText", "resizeImage", "repeat", "createPattern", "createLinearGradient", "addColorStop", "ownStacking", "testRangeBounds", "testCORS", "testSVG", "boundtest", "123px", "block", 
"selectNode", "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'></svg>", "toDataURL", "createCanvas", "loadSVGFromString", "fabric", "inlineFormatting", "hasFabric", "html2canvas.svg.js is not loaded, cannot render svg", "removeContentType", "c", "lowerCanvasEl", "renderAll", "groupSVGElements", "util", "setHeight", "setWidth", "atob", "data:image/svg+xml,", "serializeToString", "parseSVGDocument", "textTransform", "lowercase", "capitalize", "uppercase", "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7", 
"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", "offsetParent", "offsetLeft", " \r\n\t", '"', "(", "linear", "RADIAL", "GET", "status", "responseText", "statusText", "Network Error", "send", "Cannot find module '", "'", "code", "MODULE_NOT_FOUND", "MTLLoader", "manager", "DefaultLoadingManager", "EventDispatcher", "path", "setPath", "load", "texturePath", "THREE.MTLLoader: .setBaseUrl() is deprecated. Use .setTexturePath( path ) for texture path or .setPath( path ) for general base path instead.", 
"warn", "setTexturePath", "materialOptions", "charAt", "#", "newmtl", "ka", "kd", "ks", "setCrossOrigin", "setManager", "setMaterials", "MaterialCreator", "baseUrl", "materialsInfo", "materials", "materialsArray", "nameLookup", "FrontSide", "wrap", "RepeatWrapping", "convert", "normalizeRGB", "ignoreZeroRGBs", "createMaterial_", "getTextureParams", "loadTexture", "scale", "offset", "wrapS", "wrapT", "specular", "map_kd", "specularMap", "map_ks", "map_bump", "bumpMap", "bump", "shininess", "ns", "d", 
"Tr", "-bm", "bumpScale", "-s", "-o", "Handlers", "Loader", "mapping", "OBJLoader", "regexp", "fromDeclaration", "name", "currentMaterial", "_finalize", "inherited", "groupCount", "index", "smooth", "groupEnd", "number", "mtllib", "vertices", "geometry", "groupStart", "objects", "normals", "uvs", "parseVertexIndex", "addVertex", "parseUVIndex", "addUV", "parseNormalIndex", "addNormal", "Line", "addVertexLine", "addUVLine", "startObject", "time", "\r\n", "\\\n", "trimLeft", "v", "exec", "vertex_pattern", 
"n", "normal_pattern", "t", "uv_pattern", "Unexpected vertex/normal/uv line: '", "f", "face_vertex_uv_normal", "addFace", "face_vertex_uv", "face_vertex_normal", "face_vertex", "Unexpected face line: '", "l", "/", "addLineGeometry", "object_pattern", "material_use_pattern", "materialLibraries", "startMaterial", "material_library_pattern", "smoothing_pattern", "1", "on", "\x00", "Unexpected line: '", "finalize", "addAttribute", "computeVertexNormals", "uv", "LineBasicMaterial", "shading", "SmoothShading", 
"FlatShading", "addGroup", "timeEnd", "constructor", "intersection", "DDSLoader", "_parser", "CompressedTextureLoader", "DXT1", "DXT3", "DXT5", "ETC1", "THREE.DDSLoader.parse: Invalid magic number in DDS header.", "THREE.DDSLoader.parse: Unsupported format, must contain a FourCC code.", "format", "RGB_S3TC_DXT1_Format", "RGBA_S3TC_DXT3_Format", "RGBA_S3TC_DXT5_Format", "RGB_ETC1_Format", "THREE.DDSLoader.parse: Unsupported FourCC code ", "mipmapCount", "isCubemap", "THREE.DDSLoader.parse: Incomplete cubemap faces", 
"mipmaps", "PVRLoader", "[THREE.PVRLoader] Unknown PVR format", "_parseV3", "header", "RGB_PVRTC_2BPPV1_Format", "RGBA_PVRTC_2BPPV1_Format", "RGB_PVRTC_4BPPV1_Format", "RGBA_PVRTC_4BPPV1_Format", "pvrtc - unsupported PVR format ", "dataPtr", "bpp", "numSurfaces", "numMipmaps", "_parseV2", "pvrtc - unknown format ", "_extract", "buffer", "appName", "navigator", "appVersion", "language", "appCodeName", "platform", "guid=", "&url=", "&type=1", "result", "POST", "http://www.wish3d.com/license/getLicenseState.action", 
"Content-type", "application/x-www-form-urlencoded", "setRequestHeader", "onreadystatechange", "readyState", "../license/getHardwareComputerID.action", "quaternion", "getPosition", "setPosition", "getQuaternion", "setQuaternion", "cameraKeys", "looptime", "vectorKeyframeTrack", "quaternionKeyframeTrack", "interpolant", "quaInterpolant", "bPause", "startTime", "pauseTime", "init", "toArray", "flytoCamera", "createInterpolant", "flyToCamera", "getAllKeys", "play", "pause", "update", "evaluate", "updateMatrixWorld", 
"createBuffer", "bindBuffer", "bufferData", "getAttribLocation", "uvOffset", "getUniformLocation", "uvScale", "rotation", "modelViewMatrix", "alphaTest", "white", "needsUpdate", "billboards", "useProgram", "initAttributes", "disableUnusedAttributes", "elements", "uniformMatrix4fv", "activeTexture", "uniform1i", "fogType", "matrixWorld", "multiplyMatrices", "z", "getScreenRect", "geoParent", "GeoLabel", "iconPath", "setNameVisble", "attributes", "itemSize", "getAttributeBuffer", "enableAttribute", 
"vertexAttribPointer", "uniform1f", "decompose", "uniform2f", "uniform3f", "uniform2fv", "blendEquation", "blendSrc", "blendDst", "setBlending", "depthTest", "setDepthTest", "depthWrite", "setDepthWrite", "setTexture", "drawElements", "resetGLState", "createProgram", "createShader", "precision ", "getPrecision", " float;", "uniform mat4 modelViewMatrix;", "uniform mat4 projectionMatrix;", "uniform float rotation;", "uniform vec2 scale;", "uniform vec2 uvOffset;", "uniform vec2 uvScale;", "attribute vec2 position;", 
"attribute vec2 uv;", "varying vec2 vUV;", "vUV = uvOffset + uv * uvScale;", "vec2 alignedPosition = position * scale;", "vec2 rotatedPosition;", "rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;", "rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;", "vec4 finalPosition;", "finalPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );", "finalPosition.xy += rotatedPosition;", "finalPosition = projectionMatrix * finalPosition;", 
"gl_Position = finalPosition;", "shaderSource", "uniform vec3 color;", "uniform sampler2D map;", "uniform float alphaTest;", "vec4 texture = texture2D( map, vUV );", "if ( texture.a < alphaTest ) discard;", "gl_FragColor = vec4( color * texture.xyz, texture.a * opacity );", "compileShader", "attachShader", "linkProgram", "Object3D", "verticalOrign", "horizontalOrigin", "vA", "vB", "vC", "vD", "setIndex", "LSJBillboard", "raycast", "applyMatrix4", "intersectTriangle", "ray", "Er", "us", "ensurePowerOfTwo_", 
"isPowerOfTwo", "Math", "nextHighestPowerOfTwo_", "preDealPath", "\\", "//", "http:", "http://", "getDir", "getAbsolutePath", "./", ".\\", "../", "..\\", ":", "\\\\", "createXMLHttp", "XMLHttpRequest", "Microsoft.XMLHTTP", "\u60a8\u7684\u6d4f\u89c8\u5668\u4e0d\u652f\u6301\u89e3\u6790xml", "createXMLDom", "MSXML2.DOMDocument.5.0", "MSXML2.DOMDocument.4.0", "MSXML2.DOMDocument.3.0", "MSXML2.DOMDocument", "Microsoft.XMLDOM", "MSXML.DOMDocument", "createDocument", "Bold ", "getFontSize", "px ", "getFontName", 
"getStrokeWidth", "lineJoin", "computedWidth", "maxx", "minx", "dimensions", "ascent", "getStyle", "getOutlineColor", "strokeText", "getFillColor", "getPropertyValue", "measureText", "font-family", "font-size", "fontsize", "<br/>", "leading", "black", "descent", "tan", "DragControls", "off", "setObjects", "Scene", "activate", "mousemove", "addEventListener", "mousedown", "mouseup", "deactivate", "removeEventListener", "preventDefault", "clientX", "clientY", "setFromCamera", "dot", "no or infinite solutions", 
"sub", "point", "drag", "intersectObjects", "cursor", "pointer", "hoveron", "hoveroff", "move", "dragstart", "dragend", "layers", "meshGroup", "boundingSphere", "remove", "addLayer", "getBoundingSphere", "expandSphere", "getLayerByCaption", "caption", "getLayerByName", "getLayerByIndex", "empty", "releaseSelection", "FeatureLayer", "RenderableObject", "renderOrder", "RenderableFace", "v1", "v2", "v3", "normalModel", "vertexNormalsModel", "vertexNormalsLength", "RenderableVertex", "positionWorld", 
"positionScreen", "RenderableLine", "vertexColors", "RenderableSprite", "Projector", "projectVector", "THREE.Projector: .projectVector() is now vector.project().", "project", "unprojectVector", "THREE.Projector: .unprojectVector() is now vector.unproject().", "unproject", "pickingRay", "THREE.Projector: .pickingRay() is now raycaster.setFromCamera().", "getNormalMatrix", "w", "setFromPoints", "isIntersectionBox", "normalize", "applyMatrix3", "projectScene", "autoUpdate", "getInverse", "setFromMatrix", 
"lights", "Light", "Sprite", "frustumCulled", "intersectsObject", "setFromMatrixPosition", "applyProjection", "traverseVisible", "setObject", "BufferGeometry", "groups", "array", "pushVertex", "pushNormal", "pushUv", "count", "pushTriangle", "Geometry", "faces", "faceVertexUvs", "MeshFaceMaterial", "morphTargets", "morphTargetInfluences", "materialIndex", "checkTriangleVisibility", "checkBackfaceCulling", "BackSide", "negate", "vertexNormals", "pushLine", "LineSegments", "VertexColors", "abs", "lerp", 
"use strict", "MeshBasicMaterial", "setValues", "oldColor", "oldOpacity", "highlight", "setRGB", "linewidth", "TransformGizmo", "handles", "pickers", "planes", "activePlane", "XYZE", "YZ", "XZ", "handleGizmos", "pickerGizmos", "updateMatrix", "applyMatrix", "E", "search", "lookAt", "setFromRotationMatrix", "X", "Y", "Z", "setFromEuler", "TransformGizmoTranslate", "merge", "setActivePlane", "XY", "extractRotation", "XYZ", "TransformGizmoRotate", "sin", "makeRotationFromQuaternion", "atan2", "setFromAxisAngle", 
"multiplyQuaternions", "TransformGizmoScale", "TransformControls", "translationSnap", "rotationSnap", "space", "world", "size", "axis", "change", "mouseDown", "mouseUp", "objectChange", "touchstart", "touchmove", "mouseout", "touchend", "touchcancel", "touchleave", "attach", "detach", "setMode", "local", "dispatchEvent", "setTranslationSnap", "setRotationSnap", "setSpace", "distanceTo", "button", "changedTouches", "stopPropagation", "setFromMatrixScale", "rotate", "cross", "angleTo", "mode", "cubeMesh", 
"textures", "resource/skybox/sky_0.jpg", "resource/skybox/sky_1.jpg", "resource/skybox/sky_5.jpg", "resource/skybox/sky_4.jpg", "resource/skybox/sky_3.jpg", "resource/skybox/sky_2.jpg", "loadSkyBox", "RGBFormat", "setFrontTexture", "setBackTexture", "setUpTexture", "setDownTexture", "setRightTexture", "setLiftTexture", "onSceneResize", "aspect", "updateProjectionMatrix", "CanvasRenderingContext2D", "WebGLRenderingContext", "webgl", "experimental-webgl", "Worker", "File", "FileReader", "FileList", 
"Blob", "webgl-error-message", "monospace", "13px", "#fff", "#000", "1.5em", "400px", "5em auto 0", 'Your graphics card does not seem to support <a href="http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation" style="color:#000">WebGL</a>.<br />', 'Find out how to get it <a href="http://get.webgl.org/" style="color:#000">here</a>.', 'Your browser does not seem to support <a href="http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation" style="color:#000">WebGL</a>.<br/>', "oldie", "getWebGLErrorMessage", 
"depthTestUsed", "getType", "fontName", "STHeiti", "italic", "outlineVisible", "fillColor", "outlineColor", "strokeWidth", "setFontName", "setFontSize", "setFillColor", "setOutlineColor", "setStrokeWidth", "MarkerStyle", "iconColor", "iconSize", "iconVisible", "textVisible", "iconFixedSize", "textStyle", "setIconColor", "getIconPath", "setIconPath", "getIconScale", "iconScale", "setIconScale", "setStyle", "setTextStyle", "getTextStyle", "setIconVisible", "getIconVisible", "setTextVisible", "getTextVisible", 
"IconStyle", "getIconColor", "description", "getName", "getId", "GeoMarker", "strIconPath", "billboard", "needUpdate", "screenRect", "bIsNameVisble", "actualAspect", "setName", "postion", "getNameVisble", "controlCamera", "radius", "domElement", "controlRender", "applyQuaternion", "subVectors", "computeTowVecDist", "computeTowVecDistSquare", "computeDistFromEye", "Fs", "mulVec3Vec4", "isZeroVec2", "isZeroVec3", "computeSpherePixelSize", "computePixelSizeVector", "lengthSq", "GeoModel", "lengthComputable", 
"loaded", "total", "mtl", "preload", "castShadow", "receiveShadow", "computeBoundingSphere", "rotateX", "geometrys", "maxID", "curSendNode", "date", "lastTime", "getTime", "lastUpadeIndex", "attachObject", "removeAll", "GeoModelLOD", "addGeometry", "layer", "removeGeometryByName", "getGeometryByName", "removeGeometryByID", "getGeometryByID", "getGeometryByIndex", "GeoPolygon", "PageLOD", "nodeCount", "maxNodeCount", "strDataUrl", "loadStatus", "LS_UNLOAD", "sortNodes", "frustum", "viewPort", "matLocal", 
"matLocalInvert", "matModelView", "matVPW", "pixelSizeVector", "lastAccessFrame", "lastAccessTime", "maxHttpRequestNum", "curHttpRequestNum", "maxTexRequestNum", "curTexRequestNum", "maxNodeParseThreadNum", "curNodeParseThreadNum", "curLoadingNode", "bdSphere", "addNode", "pageLOD", "root", "getPixelSizeVector", "setPixelSizeVector", "getModelViewMatrix", "getFrustum", "getViewport", "setViewport", "setLastAccessTime", "getLastAccessTime", "setLastAccessFrame", "getLastAccessFrame", "addReleaseCount", 
"addNodeCount", "LS_LOADING", "responseXML", "ActiveXObject", "loadXML", "text/xml", "Scale", "getElementsByTagName", "Rotation", "OffsetMeters", "NodeList", "strDataPath", "LS_LOADED", "fromJson", "DataDefine", "Range", "Node", "addToDropList", "getLoadStatus", "computeNodeLevel", "findDropNode", "cleanRedundantNodes", "isGrandchildrenSafeDel", "unloadChildren", "computeFrustum", "distToEyeSquare", "imgUrl", "bImgBlobUrl", "PageLODNode", "childRanges", "bNormalRendered", "bInFrustumTestOk", "bdBox", 
"btLoadStatus", "enRangeMode", "bHasGeometry", "arryMaterials", "arryMaterialUsed", "dataBuffer", "dMemUsed", "setInFrustumTestOk", "isInFrustumTestOk", "setLoadStatus", "hasGeometry", "setHasGeometry", "revokeObjectURL", "minFilter", "magFilter", "netLoad", "responseType", "arraybuffer", "response", "script/lsjworker/LSJPWM.min.js", "onmessage", "bUrl", "imgBlob", "createObjectURL", "diffuseR", "diffuseG", "diffuseB", "Error:", "postMessage", "nodeMeshes", "verts", "matIndex", "indices", "colorPerNum", 
"checkInFrustum", "intersectsSphere", "intersectsBox", "computeDistSquare2Eye", "LS_NET_LOADED", "hasLoadingMaterial", "isAllMaterialLoaded", "calcNodeCount", "pop", "checkAllGroupChildLoaded", "RM_DISTANCE_FROM_EYE_POINT", "RM_PIXEL_SIZE_ON_SCREEN", "wireframe", "Wireframe", "splineHelperObjects", "splinePointsLength", "ARC_SEGMENTS", "NodeEdit", "transformControl", "delayHideTransform", "cancelHideTransorm", "hideTransform", "hiding", "clearTimeout", "points", "updateSplineOutline", "dragcontrols", 
"addSplineObject", "ambient", "addPoint", "removePoint", "mesh", "getPoint", "verticesNeedUpdate", "target", "delta", "theta", "FlyAroundCenter", "asin", "radToDeg", "degToRad", "order", "endPosition", "endHeading", "endPitch", "endRoll", "flyTo", "tube", "splineCamera", "binormal", "lookAhead", "toJSON", "duration", "lineString", "Tour", "\u7ebf\u8def\u4e00", "getPointAt", "parameters", "tangents", "binormals", "getTangentAt", "getLength", "isRemoved", "bClippingControl", "bDrawing", "point3ds", 
"spheres", "sphereGeometry", "addPoint3D", "setPoint3D", "clippingPlanes", "getWorldPosition", "fov", "updateDragging", "addMouseMoveListener", "buttons", "addLeftClickListener", "addDoubleClickListener", "createClippingPlanes", "light", "bTrack", "bTrackCamera", "shadowCameraLeft", "shadowCameraRight", "shadowCameraTop", "shadowCameraBottom", "shadowCameraNear", "shadowCameraFar", "shadowMapWidth", "shadowMapHeight", "shadowBias", "shadowMap", "cullFace", "CullFaceBack", "orbit", "up", "enableDamping", 
"dampingFactor", "enableZoom", "visble", "devicePixelRatio", "setPixelRatio", "autoUpdateScene", "autoClearColor", "addPass", "textures/tri_pattern.jpg", "resource/image/NaviCursor.png", "resize", "setClearAlpha", "vecZoomPos", "bZoomInertia", "nCurZoomTime", "dDeltaZoomRatio", "LEFT", "MOUSE", "RIGHT", "projectOnPlane", "vecPanDelta", "bPanInertia", "nCurPanTime", "dDeltaPanRatio", "bPitchInteria", "m_bMomentumZoom", "dZoomRadio", "vecRollPos", "m_bMomentumRoll", "dRotateAngle", "vecPitchPos", "m_bMomentumPitch", 
"dDeltaPitch", "bMomentumRoll", "bMomentumPitch", "bMomentumZoom", "dDeltaRollAngle", "bRollInertia", "nCurRollTime", "dDeltaTilt", "nCurPitchTime", "nTotalPanTime", "getInertiaRatio", "nTotalRollTime", "nTotalZoomTime", "nTotalPitchTime", "touches", "acos", "addSelectionObject", "Group", "Owner", "wheelDelta", "detail", "mousewheel", "contextmenu", "MozMousePixelScroll", "None", "Heightmap", "varying float height;", "void main() ", "{", "height = position.z;", "uniform vec2 colorRange;\n", "varying float height;\n", 
"void main()\n ", "{\n", "float halfRange = (colorRange.y - colorRange.x) * 0.5;", "float factor1 = clamp((height-colorRange.x) / halfRange, 0.0, 1.0);\n", "float factor2 = clamp((height-colorRange.x-halfRange) / halfRange, 0.0, 1.0);\n", "vec3 color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 1.0), factor1);", "color = mix(color, vec3(1.0, 0.0, 0.0), factor2);", "gl_FragColor = vec4(color, 1.0);\n", "}\n", "HeightmapWireframe", "getMomentumSpeed", "MOMENTUM_ROLL", "MOMENTUM_PITCH", "MOMENTUM_ZOOM", 
"INERTIA_PAN", "INERTIA_ROLL", "bRollInteria", "INERTIA_PITCH", "INERTIA_ZOOM", "bZoomInteria", "renderLine", "renderPoints", "renderMarker", "geometryMaker", "geometryMakerV", "geometryMakerH", "lineSegMaterial", "bUpdate", "edgeLabels", "computeLineDistances", "Arial", "toFixed", "\u7c73", "setBorderColor", "setBackgroundColor", "setText", "getScene", "minDistance", "maxDistance", "minZoom", "maxZoom", "minPolarAngle", "maxPolarAngle", "minAzimuthAngle", "maxAzimuthAngle", "calPhiAndThetaAngle", 
"setFromUnitVectors", "getPolarAngle", "getAzimuthalAngle", "rotateLeft", "rotateUp", "panLeft", "panUp", "pan", "PerspectiveCamera", "OrthographicCamera", "WARNING: OrbitControls.js encountered an unknown camera type - pan disabled.", "dollyIn", "zoom", "WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled.", "dollyOut", "applyAxisAngle", "distanceToSquared", "OrbitControls", "constraint", "defineProperty", "zoomSpeed", "enableRotate", "rotateSpeed", "enablePan", "keyPanSpeed", 
"autoRotate", "autoRotateSpeed", "enableKeys", "keys", "mouseButtons", "MIDDLE", "NONE", "target0", "position0", "zoom0", "reset", "pow", "ORBIT", "ROTATE", "ZOOM", "PAN", "DOLLY", "UP", "BOTTOM", "keyCode", "TOUCH_ROTATE", "pageX", "pageY", "TOUCH_DOLLY", "TOUCH_PAN", "keydown", "THREE.OrbitControls: target is now immutable. Use target.set() instead.", "THREE.OrbitControls: .noZoom has been deprecated. Use .enableZoom instead.", "THREE.OrbitControls: .noRotate has been deprecated. Use .enableRotate instead.", 
"THREE.OrbitControls: .noPan has been deprecated. Use .enablePan instead.", "THREE.OrbitControls: .noKeys has been deprecated. Use .enableKeys instead.", "THREE.OrbitControls: .staticMoving has been deprecated. Use .enableDamping instead.", "THREE.OrbitControls: .dynamicDampingFactor has been renamed. Use .dampingFactor instead.", "defineProperties", "sprite", "borderThickness", "fontface", "borderColor", "textColor", "getSprite", "setTextColor", "roundRect", "rgba(0, 0, 0, 1.0)", "lineTo", "quadraticCurveTo", 
"bTransparent", "imgUnits", "ModelLODNode", "model", "arryTextures", "arryTextureUsed", "bMap", "matrixWorldNeedsUpdate", "setMatrixWorldNeedsUpdate", ".pvr", "loadPVRTexture", ".dds", "loadDDSTexture", "LinearMipMapLinearFilter", "MirroredRepeatWrapping", "LinearMipMapNearestFilter", "anisotropy", "onprogress", "script/lsjworker/LSJLBMLoadWorker.js", "arryImages", "diffuse", "emission", "getRotate", "getScale", "compose", "rootNode", "union", "primitiveSets", "imgIndex", "isAllTextureLoaded", "hasLoadingTexture", 
"bUpadteMtl", "bSelect", "strModelPath", "setModelPath", "getModelPath", "setRotate", "setScale", "setSelect", "getBoundingBox", "selecttiongeometrys", "Features", "Feature", "Location", "Model", "Link", "pointSize", "sizeType", "Fixed", "pointSizeType", "FIXED", "PointSizeType", "pointColorType", "quality", "Squares", "heightMin", "heightMax", "setPointSize", "getPointSize", "setPointSizing", "Attenuated", "ATTENUATED", "Adaptive", "ADAPTIVE", "getPointSizing", "setQuality", "Interpolation", "isSupported", 
"SHADER_INTERPOLATION", "Splats", "SHADER_SPLATS", "getQuality", "setPointColorType", "toMaterialID", "getPointColorType", "toMaterialName", "RGB", "PointColorType", "COLOR", "Elevation", "HEIGHT", "Intensity", "INTENSITY", "Intensity Gradient", "INTENSITY_GRADIENT", "Classification", "CLASSIFICATION", "Return Number", "RETURN_NUMBER", "Source", "SOURCE", "Tree Depth", "TREE_DEPTH", "Point Index", "POINT_INDEX", "Normal", "NORMAL", "Phong", "PHONG", "setHeightRange", "getHeightRange", "pointcloud", 
"PointShape", "CIRCLE", "POCLoader", "labelSize", "element", "setVerticalOrign", "getVerticalOrign", "getHorizontalOrigin", "setHorizontalOrigin", "setLabelElement", "loading"];
THREE[FXAAShader] = {
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "resolution" : {
      value : new THREE.Vector2(1 / 1024, 1 / 512)
    }
  },
  vertexShader : [varying vec2 vUv;, void main() {, vUv = uv;, gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );, }][join](
),
  fragmentShader : [uniform sampler2D tDiffuse;, uniform vec2 resolution;, varying vec2 vUv;, #define FXAA_REDUCE_MIN   (1.0/128.0), #define FXAA_REDUCE_MUL   (1.0/8.0), #define FXAA_SPAN_MAX     8.0, void main() {, vec3 rgbNW = texture2D( tDiffuse, ( vUv - resolution )).xyz;, vec3 rgbNE = texture2D( tDiffuse, ( vUv + vec2( resolution.x, -resolution.y ) )).xyz;, vec3 rgbSW = texture2D( tDiffuse, ( vUv + vec2( -resolution.x, resolution.y ) )).xyz;, vec3 rgbSE = texture2D( tDiffuse, ( vUv + resolution )).xyz;, vec4 rgbaM  = texture2D( tDiffuse,  vUv );, vec3 rgbM  = rgbaM.xyz;, vec3 luma = vec3( 0.299, 0.587, 0.114 );, float lumaNW = dot( rgbNW, luma );, float lumaNE = dot( rgbNE, luma );, float lumaSW = dot( rgbSW, luma );, float lumaSE = dot( rgbSE, luma );, float lumaM  = dot( rgbM,  luma );, float lumaMin = min( lumaM, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );, float lumaMax = max( lumaM, max( max( lumaNW, lumaNE) , max( lumaSW, lumaSE ) ) );, vec2 dir;, dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));, dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));, float dirReduce = max( ( lumaNW + lumaNE + lumaSW + lumaSE ) * ( 0.25 * FXAA_REDUCE_MUL ), FXAA_REDUCE_MIN );, float rcpDirMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + dirReduce );, dir = min( vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),, max( vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),, dir * rcpDirMin)) * resolution;, vec4 rgbA = (1.0/2.0) * (, texture2D(tDiffuse,  vUv + dir * (1.0/3.0 - 0.5)) +, texture2D(tDiffuse,  vUv+ dir * (2.0/3.0 - 0.5)));, vec4 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (, texture2D(tDiffuse,  vUv + dir * (0.0/3.0 - 0.5)) +, texture2D(tDiffuse,  vUv + dir * (3.0/3.0 - 0.5)));, float lumaB = dot(rgbB, vec4(luma, 0.0));, if ( ( lumaB < lumaMin ) || ( lumaB > lumaMax ) ) {, gl_FragColor = rgbA;, 
  } else {, gl_FragColor = rgbB;, }, }][join](
)
};
/** @type {!Array} */
THREE[SMAAShader] = [{
  defines : {
    "SMAA_THRESHOLD" : 0.1
  },
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "resolution" : {
      value : new THREE.Vector2(1 / 1024, 1 / 512)
    }
  },
  vertexShader : [uniform vec2 resolution;, varying vec2 vUv;, varying vec4 vOffset[ 3 ];, void SMAAEdgeDetectionVS( vec2 texcoord ) {, vOffset[ 0 ] = texcoord.xyxy + resolution.xyxy * vec4( -1.0, 0.0, 0.0,  1.0 );, vOffset[ 1 ] = texcoord.xyxy + resolution.xyxy * vec4(  1.0, 0.0, 0.0, -1.0 );, vOffset[ 2 ] = texcoord.xyxy + resolution.xyxy * vec4( -2.0, 0.0, 0.0,  2.0 );, }, void main() {, vUv = uv;, SMAAEdgeDetectionVS( vUv );, gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );, }][join](
),
  fragmentShader : [uniform sampler2D tDiffuse;, varying vec2 vUv;, varying vec4 vOffset[ 3 ];, vec4 SMAAColorEdgeDetectionPS( vec2 texcoord, vec4 offset[3], sampler2D colorTex ) {, vec2 threshold = vec2( SMAA_THRESHOLD, SMAA_THRESHOLD );, vec4 delta;, vec3 C = texture2D( colorTex, texcoord ).rgb;, vec3 Cleft = texture2D( colorTex, offset[0].xy ).rgb;, vec3 t = abs( C - Cleft );, delta.x = max( max( t.r, t.g ), t.b );, vec3 Ctop = texture2D( colorTex, offset[0].zw ).rgb;, t = abs( C - Ctop );, delta.y = max( max( t.r, t.g ), t.b );, vec2 edges = step( threshold, delta.xy );, if ( dot( edges, vec2( 1.0, 1.0 ) ) == 0.0 ), discard;, vec3 Cright = texture2D( colorTex, offset[1].xy ).rgb;, t = abs( C - Cright );, delta.z = max( max( t.r, t.g ), t.b );, vec3 Cbottom  = texture2D( colorTex, offset[1].zw ).rgb;, t = abs( C - Cbottom );, delta.w = max( max( t.r, t.g ), t.b );, float maxDelta = max( max( max( delta.x, delta.y ), delta.z ), delta.w );, vec3 Cleftleft  = texture2D( colorTex, offset[2].xy ).rgb;, t = abs( C - Cleftleft );, delta.z = max( max( t.r, t.g ), t.b );, vec3 Ctoptop = texture2D( colorTex, offset[2].zw ).rgb;, t = abs( C - Ctoptop );, delta.w = max( max( t.r, t.g ), t.b );, maxDelta = max( max( maxDelta, delta.z ), delta.w );, edges.xy *= step( 0.5 * maxDelta, delta.xy );, return vec4( edges, 0.0, 0.0 );, }, void main() {, gl_FragColor = SMAAColorEdgeDetectionPS( vUv, vOffset, tDiffuse );, }][join](
)
}, {
  defines : {
    "SMAA_MAX_SEARCH_STEPS" : 8,
    "SMAA_AREATEX_MAX_DISTANCE" : 16,
    "SMAA_AREATEX_PIXEL_SIZE" : ( 1.0 / vec2( 160.0, 560.0 ) ),
    "SMAA_AREATEX_SUBTEX_SIZE" : ( 1.0 / 7.0 )
  },
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "tArea" : {
      value : null
    },
    "tSearch" : {
      value : null
    },
    "resolution" : {
      value : new THREE.Vector2(1 / 1024, 1 / 512)
    }
  },
  vertexShader : [uniform vec2 resolution;, varying vec2 vUv;, varying vec4 vOffset[ 3 ];, varying vec2 vPixcoord;, void SMAABlendingWeightCalculationVS( vec2 texcoord ) {, vPixcoord = texcoord / resolution;, vOffset[ 0 ] = texcoord.xyxy + resolution.xyxy * vec4( -0.25, 0.125, 1.25, 0.125 );, vOffset[ 1 ] = texcoord.xyxy + resolution.xyxy * vec4( -0.125, 0.25, -0.125, -1.25 );, vOffset[ 2 ] = vec4( vOffset[ 0 ].xz, vOffset[ 1 ].yw ) + vec4( -2.0, 2.0, -2.0, 2.0 ) * resolution.xxyy * float( SMAA_MAX_SEARCH_STEPS );, }, void main() {, vUv = uv;, SMAABlendingWeightCalculationVS( vUv );, gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );, }][join](
),
  fragmentShader : [#define SMAASampleLevelZeroOffset( tex, coord, offset ) texture2D( tex, coord + float( offset ) * resolution, 0.0 ), uniform sampler2D tDiffuse;, uniform sampler2D tArea;, uniform sampler2D tSearch;, uniform vec2 resolution;, varying vec2 vUv;, varying vec4 vOffset[3];, varying vec2 vPixcoord;, vec2 round( vec2 x ) {, return sign( x ) * floor( abs( x ) + 0.5 );, }, float SMAASearchLength( sampler2D searchTex, vec2 e, float bias, float scale ) {, e.r = bias + e.r * scale;, return 255.0 * texture2D( searchTex, e, 0.0 ).r;, }, float SMAASearchXLeft( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {, vec2 e = vec2( 0.0, 1.0 );, for ( int i = 0; i < SMAA_MAX_SEARCH_STEPS; i ++ ) {, e = texture2D( edgesTex, texcoord, 0.0 ).rg;, texcoord -= vec2( 2.0, 0.0 ) * resolution;, if ( ! ( texcoord.x > end && e.g > 0.8281 && e.r == 0.0 ) ) break;, }, texcoord.x += 0.25 * resolution.x;, texcoord.x += resolution.x;, texcoord.x += 2.0 * resolution.x;, texcoord.x -= resolution.x * SMAASearchLength(searchTex, e, 0.0, 0.5);, return texcoord.x;, }, float SMAASearchXRight( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {, vec2 e = vec2( 0.0, 1.0 );, for ( int i = 0; i < SMAA_MAX_SEARCH_STEPS; i ++ ) {, e = texture2D( edgesTex, texcoord, 0.0 ).rg;, texcoord += vec2( 2.0, 0.0 ) * resolution;, if ( ! ( texcoord.x < end && e.g > 0.8281 && e.r == 0.0 ) ) break;, }, texcoord.x -= 0.25 * resolution.x;, texcoord.x -= resolution.x;, 
  texcoord.x -= 2.0 * resolution.x;, texcoord.x += resolution.x * SMAASearchLength( searchTex, e, 0.5, 0.5 );, return texcoord.x;, }, float SMAASearchYUp( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {, vec2 e = vec2( 1.0, 0.0 );, for ( int i = 0; i < SMAA_MAX_SEARCH_STEPS; i ++ ) {, e = texture2D( edgesTex, texcoord, 0.0 ).rg;, texcoord += vec2( 0.0, 2.0 ) * resolution;, if ( ! ( texcoord.y > end && e.r > 0.8281 && e.g == 0.0 ) ) break;, }, texcoord.y -= 0.25 * resolution.y;, texcoord.y -= resolution.y;, texcoord.y -= 2.0 * resolution.y;, texcoord.y += resolution.y * SMAASearchLength( searchTex, e.gr, 0.0, 0.5 );, return texcoord.y;, }, float SMAASearchYDown( sampler2D edgesTex, sampler2D searchTex, vec2 texcoord, float end ) {, vec2 e = vec2( 1.0, 0.0 );, for ( int i = 0; i < SMAA_MAX_SEARCH_STEPS; i ++ ) {, e = texture2D( edgesTex, texcoord, 0.0 ).rg;, texcoord -= vec2( 0.0, 2.0 ) * resolution;, if ( ! ( texcoord.y < end && e.r > 0.8281 && e.g == 0.0 ) ) break;, }, texcoord.y += 0.25 * resolution.y;, texcoord.y += resolution.y;, texcoord.y += 2.0 * resolution.y;, texcoord.y -= resolution.y * SMAASearchLength( searchTex, e.gr, 0.5, 0.5 );, return texcoord.y;, }, vec2 SMAAArea( sampler2D areaTex, vec2 dist, float e1, float e2, float offset ) {, vec2 texcoord = float( SMAA_AREATEX_MAX_DISTANCE ) * round( 4.0 * vec2( e1, e2 ) ) + dist;, texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord + ( 0.5 * SMAA_AREATEX_PIXEL_SIZE );, texcoord.y += SMAA_AREATEX_SUBTEX_SIZE * offset;, return texture2D( areaTex, texcoord, 0.0 ).rg;, }, vec4 SMAABlendingWeightCalculationPS( vec2 texcoord, vec2 pixcoord, vec4 offset[ 3 ], sampler2D edgesTex, sampler2D areaTex, sampler2D searchTex, ivec4 subsampleIndices ) {, 
  vec4 weights = vec4( 0.0, 0.0, 0.0, 0.0 );, vec2 e = texture2D( edgesTex, texcoord ).rg;, if ( e.g > 0.0 ) {, vec2 d;, vec2 coords;, coords.x = SMAASearchXLeft( edgesTex, searchTex, offset[ 0 ].xy, offset[ 2 ].x );, coords.y = offset[ 1 ].y;, d.x = coords.x;, float e1 = texture2D( edgesTex, coords, 0.0 ).r;, coords.x = SMAASearchXRight( edgesTex, searchTex, offset[ 0 ].zw, offset[ 2 ].y );, d.y = coords.x;, d = d / resolution.x - pixcoord.x;, vec2 sqrt_d = sqrt( abs( d ) );, coords.y -= 1.0 * resolution.y;, float e2 = SMAASampleLevelZeroOffset( edgesTex, coords, ivec2( 1, 0 ) ).r;, weights.rg = SMAAArea( areaTex, sqrt_d, e1, e2, float( subsampleIndices.y ) );, }, if ( e.r > 0.0 ) {, vec2 d;, vec2 coords;, coords.y = SMAASearchYUp( edgesTex, searchTex, offset[ 1 ].xy, offset[ 2 ].z );, coords.x = offset[ 0 ].x;, d.x = coords.y;, float e1 = texture2D( edgesTex, coords, 0.0 ).g;, coords.y = SMAASearchYDown( edgesTex, searchTex, offset[ 1 ].zw, offset[ 2 ].w );, d.y = coords.y;, d = d / resolution.y - pixcoord.y;, vec2 sqrt_d = sqrt( abs( d ) );, coords.y -= 1.0 * resolution.y;, float e2 = SMAASampleLevelZeroOffset( edgesTex, coords, ivec2( 0, 1 ) ).g;, weights.ba = SMAAArea( areaTex, sqrt_d, e1, e2, float( subsampleIndices.x ) );, }, return weights;, }, void main() {, gl_FragColor = SMAABlendingWeightCalculationPS( vUv, vPixcoord, vOffset, tDiffuse, tArea, tSearch, ivec4( 0.0 ) );, }][join](
)
}, {
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "tColor" : {
      value : null
    },
    "resolution" : {
      value : new THREE.Vector2(1 / 1024, 1 / 512)
    }
  },
  vertexShader : [uniform vec2 resolution;, varying vec2 vUv;, varying vec4 vOffset[ 2 ];, void SMAANeighborhoodBlendingVS( vec2 texcoord ) {, vOffset[ 0 ] = texcoord.xyxy + resolution.xyxy * vec4( -1.0, 0.0, 0.0, 1.0 );, vOffset[ 1 ] = texcoord.xyxy + resolution.xyxy * vec4( 1.0, 0.0, 0.0, -1.0 );, }, void main() {, vUv = uv;, SMAANeighborhoodBlendingVS( vUv );, gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );, }][join](
),
  fragmentShader : [uniform sampler2D tDiffuse;, uniform sampler2D tColor;, uniform vec2 resolution;, varying vec2 vUv;, varying vec4 vOffset[ 2 ];, vec4 SMAANeighborhoodBlendingPS( vec2 texcoord, vec4 offset[ 2 ], sampler2D colorTex, sampler2D blendTex ) {, vec4 a;, a.xz = texture2D( blendTex, texcoord ).xz;, a.y = texture2D( blendTex, offset[ 1 ].zw ).g;, a.w = texture2D( blendTex, offset[ 1 ].xy ).a;, if ( dot(a, vec4( 1.0, 1.0, 1.0, 1.0 )) < 1e-5 ) {, return texture2D( colorTex, texcoord, 0.0 );, } else {, vec2 offset;, offset.x = a.a > a.b ? a.a : -a.b;, offset.y = a.g > a.r ? -a.g : a.r;, if ( abs( offset.x ) > abs( offset.y )) {, offset.y = 0.0;, } else {, offset.x = 0.0;, }, vec4 C = texture2D( colorTex, texcoord, 0.0 );, texcoord += sign( offset ) * resolution;, vec4 Cop = texture2D( colorTex, texcoord, 0.0 );, float s = abs( offset.x ) > abs( offset.y ) ? abs( offset.x ) : abs( offset.y );, C.xyz = pow(C.xyz, vec3(2.2));, Cop.xyz = pow(Cop.xyz, vec3(2.2));, vec4 mixed = mix(C, Cop, s);, mixed.xyz = pow(mixed.xyz, vec3(1.0 / 2.2));, return mixed;, }, }, void main() {, gl_FragColor = SMAANeighborhoodBlendingPS( vUv, vOffset, tColor, tDiffuse );, }][join](
)
}];
THREE[CopyShader] = {
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "opacity" : {
      value : 1
    }
  },
  vertexShader : [varying vec2 vUv;, void main() {, vUv = uv;, gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );, }][join](
),
  fragmentShader : [uniform float opacity;, uniform sampler2D tDiffuse;, varying vec2 vUv;, void main() {, vec4 texel = texture2D( tDiffuse, vUv );, gl_FragColor = opacity * texel;, }][join](
)
};
/**
 * @param {?} local$$2979
 * @param {!Array} local$$2980
 * @return {undefined}
 */
THREE[EffectComposer] = function(local$$2979, local$$2980) {
  this[renderer] = local$$2979;
  if (local$$2980 === undefined) {
    var local$$3002 = {
      minFilter : THREE[LinearFilter],
      magFilter : THREE[LinearFilter],
      format : THREE[RGBAFormat],
      stencilBuffer : false
    };
    var local$$3008 = local$$2979[getSize]();
    local$$2980 = new THREE.WebGLRenderTarget(local$$3008[width], local$$3008[height], local$$3002);
  }
  /** @type {!Array} */
  this[renderTarget1] = local$$2980;
  this[renderTarget2] = local$$2980[clone]();
  this[writeBuffer] = this[renderTarget1];
  this[readBuffer] = this[renderTarget2];
  /** @type {!Array} */
  this[passes] = [];
  if (THREE[CopyShader] === undefined) {
    console[error](THREE.EffectComposer relies on THREE.CopyShader);
  }
  this[copyPass] = new THREE.ShaderPass(THREE.CopyShader);
};
Object[assign](THREE[EffectComposer][prototype], {
  swapBuffers : function() {
    var local$$3101 = this[readBuffer];
    this[readBuffer] = this[writeBuffer];
    this[writeBuffer] = local$$3101;
  },
  addPass : function(local$$3118) {
    this[passes][push](local$$3118);
    var local$$3135 = this[renderer][getSize]();
    local$$3118[setSize](local$$3135[width], local$$3135[height]);
  },
  insertPass : function(local$$3150, local$$3151) {
    this[passes][splice](local$$3151, 0, local$$3150);
  },
  render : function(local$$3165) {
    /** @type {boolean} */
    var local$$3168 = false;
    var local$$3170;
    var local$$3172;
    var local$$3180 = this[passes][length];
    /** @type {number} */
    local$$3172 = 0;
    for (; local$$3172 < local$$3180; local$$3172++) {
      local$$3170 = this[passes][local$$3172];
      if (local$$3170[enabled] === false) {
        continue;
      }
      local$$3170[render](this[renderer], this[writeBuffer], this[readBuffer], local$$3165, local$$3168);
      if (local$$3170[needsSwap]) {
        if (local$$3168) {
          var local$$3226 = this[renderer][context];
          local$$3226[stencilFunc](local$$3226.NOTEQUAL, 1, 4294967295);
          this[copyPass][render](this[renderer], this[writeBuffer], this[readBuffer], local$$3165);
          local$$3226[stencilFunc](local$$3226.EQUAL, 1, 4294967295);
        }
        this[swapBuffers]();
      }
      if (THREE[MaskPass] !== undefined) {
        if (local$$3170 instanceof THREE[MaskPass]) {
          /** @type {boolean} */
          local$$3168 = true;
        } else {
          if (local$$3170 instanceof THREE[ClearMaskPass]) {
            /** @type {boolean} */
            local$$3168 = false;
          }
        }
      }
    }
  },
  reset : function(local$$3303) {
    if (local$$3303 === undefined) {
      var local$$3313 = this[renderer][getSize]();
      local$$3303 = this[renderTarget1][clone]();
      local$$3303[setSize](local$$3313[width], local$$3313[height]);
    }
    this[renderTarget1][dispose]();
    this[renderTarget2][dispose]();
    /** @type {!Array} */
    this[renderTarget1] = local$$3303;
    this[renderTarget2] = local$$3303[clone]();
    this[writeBuffer] = this[renderTarget1];
    this[readBuffer] = this[renderTarget2];
  },
  setSize : function(local$$3386, local$$3387) {
    this[renderTarget1][setSize](local$$3386, local$$3387);
    this[renderTarget2][setSize](local$$3386, local$$3387);
    /** @type {number} */
    var local$$3406 = 0;
    for (; local$$3406 < this[passes][length]; local$$3406++) {
      this[passes][local$$3406][setSize](local$$3386, local$$3387);
    }
  }
});
/**
 * @return {undefined}
 */
THREE[Pass] = function() {
  /** @type {boolean} */
  this[enabled] = true;
  /** @type {boolean} */
  this[needsSwap] = true;
  /** @type {boolean} */
  this[clear] = false;
  /** @type {boolean} */
  this[renderToScreen] = false;
};
Object[assign](THREE[Pass][prototype], {
  setSize : function(local$$3474, local$$3475) {
  },
  render : function(local$$3479, local$$3480, local$$3481, local$$3482, local$$3483) {
    console[error](THREE.Pass: .render() must be implemented in derived pass.);
  }
});
/**
 * @param {?} local$$3502
 * @param {?} local$$3503
 * @return {undefined}
 */
THREE[MaskPass] = function(local$$3502, local$$3503) {
  THREE[Pass][call](this);
  this[scene] = local$$3502;
  this[camera] = local$$3503;
  /** @type {boolean} */
  this[clear] = true;
  /** @type {boolean} */
  this[needsSwap] = false;
  /** @type {boolean} */
  this[inverse] = false;
};
THREE[MaskPass][prototype] = Object[assign](Object[create](THREE[Pass][prototype]), {
  constructor : THREE[MaskPass],
  render : function(local$$3567, local$$3568, local$$3569, local$$3570, local$$3571) {
    var local$$3576 = local$$3567[context];
    var local$$3581 = local$$3567[state];
    local$$3581[buffers][color][setMask](false);
    local$$3581[buffers][depth][setMask](false);
    local$$3581[buffers][color][setLocked](true);
    local$$3581[buffers][depth][setLocked](true);
    var local$$3631;
    var local$$3633;
    if (this[inverse]) {
      /** @type {number} */
      local$$3631 = 0;
      /** @type {number} */
      local$$3633 = 1;
    } else {
      /** @type {number} */
      local$$3631 = 1;
      /** @type {number} */
      local$$3633 = 0;
    }
    local$$3581[buffers][stencil][setTest](true);
    local$$3581[buffers][stencil][setOp](local$$3576.REPLACE, local$$3576.REPLACE, local$$3576.REPLACE);
    local$$3581[buffers][stencil][setFunc](local$$3576.ALWAYS, local$$3631, 4294967295);
    local$$3581[buffers][stencil][setClear](local$$3633);
    local$$3567[render](this[scene], this[camera], local$$3569, this[clear]);
    local$$3567[render](this[scene], this[camera], local$$3568, this[clear]);
    local$$3581[buffers][color][setLocked](false);
    local$$3581[buffers][depth][setLocked](false);
    local$$3581[buffers][stencil][setFunc](local$$3576.EQUAL, 1, 4294967295);
    local$$3581[buffers][stencil][setOp](local$$3576.KEEP, local$$3576.KEEP, local$$3576.KEEP);
  }
});
/**
 * @return {undefined}
 */
THREE[ClearMaskPass] = function() {
  THREE[Pass][call](this);
  /** @type {boolean} */
  this[needsSwap] = false;
};
THREE[ClearMaskPass][prototype] = Object[create](THREE[Pass][prototype]);
Object[assign](THREE[ClearMaskPass][prototype], {
  render : function(local$$3842, local$$3843, local$$3844, local$$3845, local$$3846) {
    local$$3842[state][buffers][stencil][setTest](false);
  }
});
/**
 * @param {?} local$$3873
 * @param {?} local$$3874
 * @param {?} local$$3875
 * @param {?} local$$3876
 * @param {number} local$$3877
 * @return {undefined}
 */
THREE[RenderPass] = function(local$$3873, local$$3874, local$$3875, local$$3876, local$$3877) {
  THREE[Pass][call](this);
  this[scene] = local$$3873;
  this[camera] = local$$3874;
  this[overrideMaterial] = local$$3875;
  this[clearColor] = local$$3876;
  this[clearAlpha] = local$$3877 !== undefined ? local$$3877 : 0;
  /** @type {boolean} */
  this[clear] = true;
  /** @type {boolean} */
  this[needsSwap] = false;
};
THREE[RenderPass][prototype] = Object[assign](Object[create](THREE[Pass][prototype]), {
  constructor : THREE[RenderPass],
  render : function(local$$3953, local$$3954, local$$3955, local$$3956, local$$3957) {
    var local$$3962 = local$$3953[autoClear];
    /** @type {boolean} */
    local$$3953[autoClear] = false;
    this[scene][overrideMaterial] = this[overrideMaterial];
    var local$$3981;
    var local$$3983;
    if (this[clearColor]) {
      local$$3981 = local$$3953[getClearColor]()[getHex]();
      local$$3983 = local$$3953[getClearAlpha]();
      local$$3953[setClearColor](this[clearColor], this[clearAlpha]);
    }
    local$$3953[render](this[scene], this[camera], this[renderToScreen] ? null : local$$3955, this[clear]);
    if (this[clearColor]) {
      local$$3953[setClearColor](local$$3981, local$$3983);
    }
    /** @type {null} */
    this[scene][overrideMaterial] = null;
    local$$3953[autoClear] = local$$3962;
  }
});
/**
 * @param {?} local$$4073
 * @param {string} local$$4074
 * @return {undefined}
 */
THREE[ShaderPass] = function(local$$4073, local$$4074) {
  THREE[Pass][call](this);
  this[textureID] = local$$4074 !== undefined ? local$$4074 : tDiffuse;
  if (local$$4073 instanceof THREE[ShaderMaterial]) {
    this[uniforms] = local$$4073[uniforms];
    this[material] = local$$4073;
  } else {
    if (local$$4073) {
      this[uniforms] = THREE[UniformsUtils][clone](local$$4073[uniforms]);
      this[material] = new THREE.ShaderMaterial({
        defines : local$$4073[defines] || {},
        uniforms : this[uniforms],
        vertexShader : local$$4073[vertexShader],
        fragmentShader : local$$4073[fragmentShader]
      });
    }
  }
  this[camera] = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  this[scene] = new THREE.Scene;
  this[quad] = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), null);
  this[scene][add](this[quad]);
};
THREE[ShaderPass][prototype] = Object[assign](Object[create](THREE[Pass][prototype]), {
  constructor : THREE[ShaderPass],
  render : function(local$$4223, local$$4224, local$$4225, local$$4226, local$$4227) {
    if (this[uniforms][this[textureID]]) {
      this[uniforms][this[textureID]][value] = local$$4225[texture];
    }
    this[quad][material] = this[material];
    if (this[renderToScreen]) {
      local$$4223[render](this[scene], this[camera]);
    } else {
      local$$4223[render](this[scene], this[camera], local$$4224, this[clear]);
    }
  }
});
/**
 * @param {string} local$$4307
 * @param {?} local$$4308
 * @param {?} local$$4309
 * @param {number} local$$4310
 * @return {undefined}
 */
THREE[OutlinePass] = function(local$$4307, local$$4308, local$$4309, local$$4310) {
  this[renderScene] = local$$4308;
  this[renderCamera] = local$$4309;
  this[selectedObjects] = local$$4310 !== undefined ? local$$4310 : [];
  this[visibleEdgeColor] = new THREE.Color(1, 1, 1);
  this[hiddenEdgeColor] = new THREE.Color(.1, .04, .02);
  /** @type {number} */
  this[edgeGlow] = .4;
  /** @type {boolean} */
  this[usePatternTexture] = false;
  /** @type {number} */
  this[edgeThickness] = 1.4;
  /** @type {number} */
  this[edgeStrength] = 8;
  /** @type {number} */
  this[downSampleRatio] = 1;
  /** @type {number} */
  this[pulsePeriod] = 2;
  THREE[Pass][call](this);
  this[resolution] = local$$4307 !== undefined ? new THREE.Vector2(local$$4307[x], local$$4307[y]) : new THREE.Vector2(256, 256);
  var local$$4423 = {
    minFilter : THREE[LinearFilter],
    magFilter : THREE[LinearFilter],
    format : THREE[RGBAFormat]
  };
  var local$$4440 = Math[round](this[resolution][x] / this[downSampleRatio]);
  var local$$4456 = Math[round](this[resolution][y] / this[downSampleRatio]);
  this[maskBufferMaterial] = new THREE.MeshBasicMaterial({
    color : 16777215
  });
  this[maskBufferMaterial][side] = THREE[DoubleSide];
  this[renderTargetMaskBuffer] = new THREE.WebGLRenderTarget(this[resolution][x], this[resolution][y], local$$4423);
  /** @type {boolean} */
  this[renderTargetMaskBuffer][texture][generateMipmaps] = false;
  this[depthMaterial] = new THREE.MeshDepthMaterial;
  this[depthMaterial][side] = THREE[DoubleSide];
  this[depthMaterial][depthPacking] = THREE[RGBADepthPacking];
  this[depthMaterial][blending] = THREE[NoBlending];
  this[prepareMaskMaterial] = this[getPrepareMaskMaterial]();
  this[prepareMaskMaterial][side] = THREE[DoubleSide];
  this[renderTargetDepthBuffer] = new THREE.WebGLRenderTarget(this[resolution][x], this[resolution][y], local$$4423);
  /** @type {boolean} */
  this[renderTargetDepthBuffer][texture][generateMipmaps] = false;
  this[renderTargetMaskDownSampleBuffer] = new THREE.WebGLRenderTarget(local$$4440, local$$4456, local$$4423);
  /** @type {boolean} */
  this[renderTargetMaskDownSampleBuffer][texture][generateMipmaps] = false;
  this[renderTargetBlurBuffer1] = new THREE.WebGLRenderTarget(local$$4440, local$$4456, local$$4423);
  /** @type {boolean} */
  this[renderTargetBlurBuffer1][texture][generateMipmaps] = false;
  this[renderTargetBlurBuffer2] = new THREE.WebGLRenderTarget(Math[round](local$$4440 / 2), Math[round](local$$4456 / 2), local$$4423);
  /** @type {boolean} */
  this[renderTargetBlurBuffer2][texture][generateMipmaps] = false;
  this[edgeDetectionMaterial] = this[getEdgeDetectionMaterial]();
  this[renderTargetEdgeBuffer1] = new THREE.WebGLRenderTarget(local$$4440, local$$4456, local$$4423);
  /** @type {boolean} */
  this[renderTargetEdgeBuffer1][texture][generateMipmaps] = false;
  this[renderTargetEdgeBuffer2] = new THREE.WebGLRenderTarget(Math[round](local$$4440 / 2), Math[round](local$$4456 / 2), local$$4423);
  /** @type {boolean} */
  this[renderTargetEdgeBuffer2][texture][generateMipmaps] = false;
  /** @type {number} */
  var local$$4730 = 4;
  /** @type {number} */
  var local$$4733 = 4;
  this[separableBlurMaterial1] = this[getSeperableBlurMaterial](local$$4730);
  this[separableBlurMaterial1][uniforms][texSize][value] = new THREE.Vector2(local$$4440, local$$4456);
  /** @type {number} */
  this[separableBlurMaterial1][uniforms][kernelRadius][value] = 1;
  this[separableBlurMaterial2] = this[getSeperableBlurMaterial](local$$4733);
  this[separableBlurMaterial2][uniforms][texSize][value] = new THREE.Vector2(Math[round](local$$4440 / 2), Math[round](local$$4456 / 2));
  /** @type {number} */
  this[separableBlurMaterial2][uniforms][kernelRadius][value] = local$$4733;
  this[overlayMaterial] = this[getOverlayMaterial]();
  if (THREE[CopyShader] === undefined) {
    console[error](THREE.OutlinePass relies on THREE.CopyShader);
  }
  var local$$4852 = THREE[CopyShader];
  this[copyUniforms] = THREE[UniformsUtils][clone](local$$4852[uniforms]);
  /** @type {number} */
  this[copyUniforms][opacity][value] = 1;
  this[materialCopy] = new THREE.ShaderMaterial({
    uniforms : this[copyUniforms],
    vertexShader : local$$4852[vertexShader],
    fragmentShader : local$$4852[fragmentShader],
    blending : THREE[NoBlending],
    depthTest : false,
    depthWrite : false,
    transparent : true
  });
  /** @type {boolean} */
  this[enabled] = true;
  /** @type {boolean} */
  this[needsSwap] = false;
  this[oldClearColor] = new THREE.Color;
  /** @type {number} */
  this[oldClearAlpha] = 1;
  this[camera] = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  this[scene] = new THREE.Scene;
  this[quad] = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), null);
  this[scene][add](this[quad]);
  this[tempPulseColor1] = new THREE.Color;
  this[tempPulseColor2] = new THREE.Color;
  this[textureMatrix] = new THREE.Matrix4;
};
THREE[OutlinePass][prototype] = Object[assign](Object[create](THREE[Pass][prototype]), {
  constructor : THREE[OutlinePass],
  dispose : function() {
    this[renderTargetMaskBuffer][dispose]();
    this[renderTargetDepthBuffer][dispose]();
    this[renderTargetMaskDownSampleBuffer][dispose]();
    this[renderTargetBlurBuffer1][dispose]();
    this[renderTargetBlurBuffer2][dispose]();
    this[renderTargetEdgeBuffer1][dispose]();
    this[renderTargetEdgeBuffer2][dispose]();
  },
  setSize : function(local$$5079, local$$5080) {
    this[renderTargetMaskBuffer][setSize](local$$5079, local$$5080);
    var local$$5098 = Math[round](local$$5079 / this[downSampleRatio]);
    var local$$5108 = Math[round](local$$5080 / this[downSampleRatio]);
    this[renderTargetMaskDownSampleBuffer][setSize](local$$5098, local$$5108);
    this[renderTargetBlurBuffer1][setSize](local$$5098, local$$5108);
    this[renderTargetEdgeBuffer1][setSize](local$$5098, local$$5108);
    this[separableBlurMaterial1][uniforms][texSize][value] = new THREE.Vector2(local$$5098, local$$5108);
    local$$5098 = Math[round](local$$5098 / 2);
    local$$5108 = Math[round](local$$5108 / 2);
    this[renderTargetBlurBuffer2][setSize](local$$5098, local$$5108);
    this[renderTargetEdgeBuffer2][setSize](local$$5098, local$$5108);
    this[separableBlurMaterial2][uniforms][texSize][value] = new THREE.Vector2(local$$5098, local$$5108);
  },
  changeVisibilityOfSelectedObjects : function(local$$5200) {
    /**
     * @param {?} local$$5202
     * @return {undefined}
     */
    var local$$5217 = function(local$$5202) {
      if (local$$5202 instanceof THREE[Mesh]) {
        local$$5202[visible] = local$$5200;
      }
    };
    /** @type {number} */
    var local$$5220 = 0;
    for (; local$$5220 < this[selectedObjects][length]; local$$5220++) {
      var local$$5235 = this[selectedObjects][local$$5220];
      local$$5235[traverse](local$$5217);
    }
  },
  changeVisibilityOfNonSelectedObjects : function(local$$5246) {
    /** @type {!Array} */
    var local$$5249 = [];
    /**
     * @param {?} local$$5251
     * @return {undefined}
     */
    var local$$5266 = function(local$$5251) {
      if (local$$5251 instanceof THREE[Mesh]) {
        local$$5249[push](local$$5251);
      }
    };
    /** @type {number} */
    var local$$5269 = 0;
    for (; local$$5269 < this[selectedObjects][length]; local$$5269++) {
      var local$$5284 = this[selectedObjects][local$$5269];
      local$$5284[traverse](local$$5266);
    }
    /**
     * @param {?} local$$5294
     * @return {undefined}
     */
    var local$$5361 = function(local$$5294) {
      if (local$$5294 instanceof THREE[Mesh]) {
        /** @type {boolean} */
        var local$$5301 = false;
        /** @type {number} */
        var local$$5304 = 0;
        for (; local$$5304 < local$$5249[length]; local$$5304++) {
          var local$$5316 = local$$5249[local$$5304][id];
          if (local$$5316 === local$$5294[id]) {
            /** @type {boolean} */
            local$$5301 = true;
            break;
          }
        }
        if (!local$$5301) {
          var local$$5335 = local$$5294[visible];
          if (!local$$5246 || local$$5294[bVisible]) {
            local$$5294[visible] = local$$5246;
          }
          local$$5294[bVisible] = local$$5335;
        }
      }
    };
    this[renderScene][traverse](local$$5361);
  },
  updateTextureMatrix : function() {
    this[textureMatrix][set](.5, 0, 0, .5, 0, .5, 0, .5, 0, 0, .5, .5, 0, 0, 0, 1);
    this[textureMatrix][multiply](this[renderCamera][projectionMatrix]);
    this[textureMatrix][multiply](this[renderCamera][matrixWorldInverse]);
  },
  render : function(local$$5428, local$$5429, local$$5430, local$$5431, local$$5432) {
    if (this[selectedObjects][length] === 0) {
      return;
    }
    this[oldClearColor][copy](local$$5428[getClearColor]());
    this[oldClearAlpha] = local$$5428[getClearAlpha]();
    var local$$5470 = local$$5428[autoClear];
    /** @type {boolean} */
    local$$5428[autoClear] = false;
    if (local$$5432) {
      local$$5428[context][disable](local$$5428[context].STENCIL_TEST);
    }
    local$$5428[setClearColor](16777215, 1);
    this[changeVisibilityOfSelectedObjects](false);
    this[renderScene][overrideMaterial] = this[depthMaterial];
    local$$5428[render](this[renderScene], this[renderCamera], this[renderTargetDepthBuffer], true);
    this[changeVisibilityOfSelectedObjects](true);
    this[updateTextureMatrix]();
    this[changeVisibilityOfNonSelectedObjects](false);
    var local$$5556 = this[renderScene][background];
    /** @type {null} */
    this[renderScene][background] = null;
    this[renderScene][overrideMaterial] = this[prepareMaskMaterial];
    this[prepareMaskMaterial][uniforms][cameraNearFar][value] = new THREE.Vector2(this[renderCamera][near], this[renderCamera][far]);
    this[prepareMaskMaterial][uniforms][depthTexture][value] = this[renderTargetDepthBuffer][texture];
    this[prepareMaskMaterial][uniforms][textureMatrix][value] = this[textureMatrix];
    local$$5428[render](this[renderScene], this[renderCamera], this[renderTargetMaskBuffer], true);
    /** @type {null} */
    this[renderScene][overrideMaterial] = null;
    this[changeVisibilityOfNonSelectedObjects](true);
    this[renderScene][background] = local$$5556;
    this[quad][material] = this[materialCopy];
    this[copyUniforms][tDiffuse][value] = this[renderTargetMaskBuffer][texture];
    local$$5428[render](this[scene], this[camera], this[renderTargetMaskDownSampleBuffer], true);
    this[tempPulseColor1][copy](this[visibleEdgeColor]);
    this[tempPulseColor2][copy](this[hiddenEdgeColor]);
    if (this[pulsePeriod] > 0) {
      /** @type {number} */
      var local$$5778 = (1 + .25) / 2 + Math[cos](performance[now]() * .01 / this[pulsePeriod]) * (1 - .25) / 2;
      this[tempPulseColor1][multiplyScalar](local$$5778);
      this[tempPulseColor2][multiplyScalar](local$$5778);
    }
    this[quad][material] = this[edgeDetectionMaterial];
    this[edgeDetectionMaterial][uniforms][maskTexture][value] = this[renderTargetMaskDownSampleBuffer][texture];
    this[edgeDetectionMaterial][uniforms][texSize][value] = new THREE.Vector2(this[renderTargetMaskDownSampleBuffer][width], this[renderTargetMaskDownSampleBuffer][height]);
    this[edgeDetectionMaterial][uniforms][visibleEdgeColor][value] = this[tempPulseColor1];
    this[edgeDetectionMaterial][uniforms][hiddenEdgeColor][value] = this[tempPulseColor2];
    local$$5428[render](this[scene], this[camera], this[renderTargetEdgeBuffer1], true);
    this[quad][material] = this[separableBlurMaterial1];
    this[separableBlurMaterial1][uniforms][colorTexture][value] = this[renderTargetEdgeBuffer1][texture];
    this[separableBlurMaterial1][uniforms][direction][value] = THREE[OutlinePass][BlurDirectionX];
    this[separableBlurMaterial1][uniforms][kernelRadius][value] = this[edgeThickness];
    local$$5428[render](this[scene], this[camera], this[renderTargetBlurBuffer1], true);
    this[separableBlurMaterial1][uniforms][colorTexture][value] = this[renderTargetBlurBuffer1][texture];
    this[separableBlurMaterial1][uniforms][direction][value] = THREE[OutlinePass][BlurDirectionY];
    local$$5428[render](this[scene], this[camera], this[renderTargetEdgeBuffer1], true);
    this[quad][material] = this[separableBlurMaterial2];
    this[separableBlurMaterial2][uniforms][colorTexture][value] = this[renderTargetEdgeBuffer1][texture];
    this[separableBlurMaterial2][uniforms][direction][value] = THREE[OutlinePass][BlurDirectionX];
    local$$5428[render](this[scene], this[camera], this[renderTargetBlurBuffer2], true);
    this[separableBlurMaterial2][uniforms][colorTexture][value] = this[renderTargetBlurBuffer2][texture];
    this[separableBlurMaterial2][uniforms][direction][value] = THREE[OutlinePass][BlurDirectionY];
    local$$5428[render](this[scene], this[camera], this[renderTargetEdgeBuffer2], true);
    this[quad][material] = this[overlayMaterial];
    this[overlayMaterial][uniforms][maskTexture][value] = this[renderTargetMaskBuffer][texture];
    this[overlayMaterial][uniforms][edgeTexture1][value] = this[renderTargetEdgeBuffer1][texture];
    this[overlayMaterial][uniforms][edgeTexture2][value] = this[renderTargetEdgeBuffer2][texture];
    this[overlayMaterial][uniforms][patternTexture][value] = this[patternTexture];
    this[overlayMaterial][uniforms][edgeStrength][value] = this[edgeStrength];
    this[overlayMaterial][uniforms][edgeGlow][value] = this[edgeGlow];
    this[overlayMaterial][uniforms][usePatternTexture][value] = this[usePatternTexture];
    if (local$$5432) {
      local$$5428[context][enable](local$$5428[context].STENCIL_TEST);
    }
    local$$5428[render](this[scene], this[camera], local$$5430, false);
    local$$5428[setClearColor](this[oldClearColor], this[oldClearAlpha]);
    local$$5428[autoClear] = local$$5470;
  },
  getPrepareMaskMaterial : function() {
    return new THREE.ShaderMaterial({
      uniforms : {
        "depthTexture" : {
          value : null
        },
        "cameraNearFar" : {
          value : new THREE.Vector2(.5, .5)
        },
        "textureMatrix" : {
          value : new THREE.Matrix4
        }
      },
      vertexShader : varying vec2 vUv;
				varying vec4 projTexCoord;
				varying vec4 vPosition;
				uniform mat4 textureMatrix;
				void main() {
					vUv = uv;
					vPosition = modelViewMatrix * vec4( position, 1.0 );
					vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
					projTexCoord = textureMatrix * worldPosition;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

				},
      fragmentShader : #include <packing>
				varying vec2 vUv;
				varying vec4 vPosition;
				varying vec4 projTexCoord;
				uniform sampler2D depthTexture;
				uniform vec2 cameraNearFar;
				
				void main() {
					float depth = unpackRGBAToDepth(texture2DProj( depthTexture, projTexCoord ));
					float viewZ = -perspectiveDepthToViewZ( depth, cameraNearFar.x, cameraNearFar.y );
					float depthTest = (-vPosition.z > viewZ) ? 1.0 : 0.0;
					gl_FragColor = vec4(0.0, depthTest, 1.0, 1.0);
				}
    });
  },
  getEdgeDetectionMaterial : function() {
    return new THREE.ShaderMaterial({
      uniforms : {
        "maskTexture" : {
          value : null
        },
        "texSize" : {
          value : new THREE.Vector2(.5, .5)
        },
        "visibleEdgeColor" : {
          value : new THREE.Vector3(1, 1, 1)
        },
        "hiddenEdgeColor" : {
          value : new THREE.Vector3(1, 1, 1)
        }
      },
      vertexShader : varying vec2 vUv;

				void main() {

					vUv = uv;

					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

				},
      fragmentShader : varying vec2 vUv;
				uniform sampler2D maskTexture;
				uniform vec2 texSize;
				uniform vec3 visibleEdgeColor;
				uniform vec3 hiddenEdgeColor;
				
				void main() {

					vec2 invSize = 1.0 / texSize;
					vec4 uvOffset = vec4(1.0, 0.0, 0.0, 1.0) * vec4(invSize, invSize);
					vec4 c1 = texture2D( maskTexture, vUv + uvOffset.xy);
					vec4 c2 = texture2D( maskTexture, vUv - uvOffset.xy);
					vec4 c3 = texture2D( maskTexture, vUv + uvOffset.yw);
					vec4 c4 = texture2D( maskTexture, vUv - uvOffset.yw);
					float diff1 = (c1.r - c2.r)*0.5;
					float diff2 = (c3.r - c4.r)*0.5;
					float d = length( vec2(diff1, diff2) );
					float a1 = min(c1.g, c2.g);
					float a2 = min(c3.g, c4.g);
					float visibilityFactor = min(a1, a2);
					vec3 edgeColor = 1.0 - visibilityFactor > 0.001 ? visibleEdgeColor : hiddenEdgeColor;
					gl_FragColor = vec4(edgeColor, 1.0) * vec4(d);
				}
    });
  },
  getSeperableBlurMaterial : function(local$$6404) {
    return new THREE.ShaderMaterial({
      defines : {
        "MAX_RADIUS" : local$$6404
      },
      uniforms : {
        "colorTexture" : {
          value : null
        },
        "texSize" : {
          value : new THREE.Vector2(.5, .5)
        },
        "direction" : {
          value : new THREE.Vector2(.5, .5)
        },
        "kernelRadius" : {
          value : 1
        }
      },
      vertexShader : varying vec2 vUv;

				void main() {

					vUv = uv;

					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

				},
      fragmentShader : #include <common>
				varying vec2 vUv;
				uniform sampler2D colorTexture;
				uniform vec2 texSize;
				uniform vec2 direction;
				uniform float kernelRadius;
				
				float gaussianPdf(in float x, in float sigma) {
					return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;
				}
				void main() {
					vec2 invSize = 1.0 / texSize;
					float weightSum = gaussianPdf(0.0, kernelRadius);
					vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;
					vec2 delta = direction * invSize * kernelRadius/float(MAX_RADIUS);
					vec2 uvOffset = delta;
					for( int i = 1; i <= MAX_RADIUS; i ++ ) {
						float w = gaussianPdf(uvOffset.x, kernelRadius);
						vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;
						vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;
						diffuseSum += ((sample1 + sample2) * w);
						weightSum += (2.0 * w);
						uvOffset += delta;
					}
					gl_FragColor = vec4(diffuseSum/weightSum, 1.0);
				}
    });
  },
  getOverlayMaterial : function() {
    return new THREE.ShaderMaterial({
      uniforms : {
        "maskTexture" : {
          value : null
        },
        "edgeTexture1" : {
          value : null
        },
        "edgeTexture2" : {
          value : null
        },
        "patternTexture" : {
          value : null
        },
        "edgeStrength" : {
          value : 1
        },
        "edgeGlow" : {
          value : 1
        },
        "usePatternTexture" : {
          value : 0
        }
      },
      vertexShader : varying vec2 vUv;

				void main() {

					vUv = uv;

					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

				},
      fragmentShader : varying vec2 vUv;
				uniform sampler2D maskTexture;
				uniform sampler2D edgeTexture1;
				uniform sampler2D edgeTexture2;
				uniform sampler2D patternTexture;
				uniform float edgeStrength;
				uniform float edgeGlow;
				uniform bool usePatternTexture;
				
				void main() {
					vec4 edgeValue1 = texture2D(edgeTexture1, vUv);
					vec4 edgeValue2 = texture2D(edgeTexture2, vUv);
					vec4 maskColor = texture2D(maskTexture, vUv);
					vec4 patternColor = texture2D(patternTexture, 6.0 * vUv);
					float visibilityFactor = 1.0 - maskColor.g > 0.0 ? 1.0 : 0.5;
					vec4 edgeValue = edgeValue1 + edgeValue2 * edgeGlow;
					vec4 finalColor = edgeStrength * edgeValue;
					if(usePatternTexture)
						finalColor += + visibilityFactor * (1.0 - maskColor.r) * (1.0 - patternColor.r);
					gl_FragColor = finalColor;
				},
      blending : THREE[AdditiveBlending],
      depthTest : false,
      depthWrite : false,
      transparent : true
    });
  }
});
THREE[OutlinePass][BlurDirectionX] = new THREE.Vector2(1, 0);
THREE[OutlinePass][BlurDirectionY] = new THREE.Vector2(0, 1);
!function(local$$6495) {
  if (object == typeof exports && undefined != typeof module) {
    module[exports] = local$$6495();
  } else {
    if (function == typeof define && define[amd]) {
      define([], local$$6495);
    } else {
      var local$$6528;
      if (undefined != typeof window) {
        /** @type {!Window} */
        local$$6528 = window;
      } else {
        if (undefined != typeof global) {
          local$$6528 = global;
        } else {
          if (undefined != typeof self) {
            /** @type {!Window} */
            local$$6528 = self;
          }
        }
      }
      local$$6528[html2canvas] = local$$6495();
    }
  }
}(function() {
  var local$$6572;
  var local$$6574;
  var local$$6576;
  return function local$$6578(local$$6579, local$$6580, local$$6581) {
    /**
     * @param {?} local$$6584
     * @param {?} local$$6585
     * @return {?}
     */
    function local$$6583(local$$6584, local$$6585) {
      if (!local$$6580[local$$6584]) {
        if (!local$$6579[local$$6584]) {
          var local$$6597 = typeof require == function && require;
          if (!local$$6585 && local$$6597) {
            return local$$6597(local$$6584, true);
          }
          if (local$$6607) {
            return local$$6607(local$$6584, true);
          }
          /** @type {!Error} */
          var local$$6622 = new Error(Cannot find module ' + local$$6584 + ');
          throw local$$6622[code] = MODULE_NOT_FOUND, local$$6622;
        }
        var local$$6639 = local$$6580[local$$6584] = {
          exports : {}
        };
        local$$6579[local$$6584][0][call](local$$6639[exports], function(local$$6650) {
          var local$$6656 = local$$6579[local$$6584][1][local$$6650];
          return local$$6583(local$$6656 ? local$$6656 : local$$6650);
        }, local$$6639, local$$6639[exports], local$$6578, local$$6579, local$$6580, local$$6581);
      }
      return local$$6580[local$$6584][exports];
    }
    var local$$6607 = typeof require == function && require;
    /** @type {number} */
    var local$$6685 = 0;
    for (; local$$6685 < local$$6581[length]; local$$6685++) {
      local$$6583(local$$6581[local$$6685]);
    }
    return local$$6583;
  }({
    1 : [function(local$$6702, local$$6703, local$$6704) {
      (function(local$$6706) {
        (function(local$$6710) {
          /**
           * @param {?} local$$6713
           * @return {?}
           */
          function local$$6712(local$$6713) {
            throw RangeError(local$$6716[local$$6713]);
          }
          /**
           * @param {!Array} local$$6723
           * @param {!Function} local$$6724
           * @return {?}
           */
          function local$$6722(local$$6723, local$$6724) {
            var local$$6729 = local$$6723[length];
            for (; local$$6729--;) {
              local$$6723[local$$6729] = local$$6724(local$$6723[local$$6729]);
            }
            return local$$6723;
          }
          /**
           * @param {?} local$$6746
           * @param {!Function} local$$6747
           * @return {?}
           */
          function local$$6745(local$$6746, local$$6747) {
            return local$$6722(local$$6746[split](local$$6752), local$$6747)[join](.);
          }
          /**
           * @param {!Array} local$$6765
           * @return {?}
           */
          function local$$6764(local$$6765) {
            /** @type {!Array} */
            var local$$6768 = [];
            /** @type {number} */
            var local$$6771 = 0;
            var local$$6776 = local$$6765[length];
            var local$$6778;
            var local$$6780;
            for (; local$$6771 < local$$6776;) {
              local$$6778 = local$$6765[charCodeAt](local$$6771++);
              if (local$$6778 >= 55296 && local$$6778 <= 56319 && local$$6771 < local$$6776) {
                local$$6780 = local$$6765[charCodeAt](local$$6771++);
                if ((local$$6780 & 64512) == 56320) {
                  local$$6768[push](((local$$6778 & 1023) << 10) + (local$$6780 & 1023) + 65536);
                } else {
                  local$$6768[push](local$$6778);
                  local$$6771--;
                }
              } else {
                local$$6768[push](local$$6778);
              }
            }
            return local$$6768;
          }
          /**
           * @param {!Array} local$$6849
           * @return {?}
           */
          function local$$6848(local$$6849) {
            return local$$6722(local$$6849, function(local$$6851) {
              var local$$6855 = ;
              if (local$$6851 > 65535) {
                /** @type {number} */
                local$$6851 = local$$6851 - 65536;
                local$$6855 = local$$6855 + local$$6863(local$$6851 >>> 10 & 1023 | 55296);
                /** @type {number} */
                local$$6851 = 56320 | local$$6851 & 1023;
              }
              local$$6855 = local$$6855 + local$$6863(local$$6851);
              return local$$6855;
            })[join]();
          }
          /**
           * @param {number} local$$6901
           * @return {?}
           */
          function local$$6900(local$$6901) {
            if (local$$6901 - 48 < 10) {
              return local$$6901 - 22;
            }
            if (local$$6901 - 65 < 26) {
              return local$$6901 - 65;
            }
            if (local$$6901 - 97 < 26) {
              return local$$6901 - 97;
            }
            return local$$6933;
          }
          /**
           * @param {number} local$$6938
           * @param {number} local$$6939
           * @return {?}
           */
          function local$$6937(local$$6938, local$$6939) {
            return local$$6938 + 22 + 75 * (local$$6938 < 26) - ((local$$6939 != 0) << 5);
          }
          /**
           * @param {number} local$$6957
           * @param {number} local$$6958
           * @param {boolean} local$$6959
           * @return {?}
           */
          function local$$6956(local$$6957, local$$6958, local$$6959) {
            /** @type {number} */
            var local$$6962 = 0;
            local$$6957 = local$$6959 ? local$$6964(local$$6957 / local$$6965) : local$$6957 >> 1;
            local$$6957 = local$$6957 + local$$6964(local$$6957 / local$$6958);
            for (; local$$6957 > local$$6979 * local$$6980 >> 1; local$$6962 = local$$6962 + local$$6933) {
              local$$6957 = local$$6964(local$$6957 / local$$6979);
            }
            return local$$6964(local$$6962 + (local$$6979 + 1) * local$$6957 / (local$$6957 + local$$6997));
          }
          /**
           * @param {!Array} local$$7006
           * @return {?}
           */
          function local$$7005(local$$7006) {
            /** @type {!Array} */
            var local$$7009 = [];
            var local$$7014 = local$$7006[length];
            var local$$7016;
            /** @type {number} */
            var local$$7019 = 0;
            /** @type {number} */
            var local$$7022 = local$$7021;
            /** @type {number} */
            var local$$7025 = local$$7024;
            var local$$7027;
            var local$$7029;
            var local$$7031;
            var local$$7033;
            var local$$7035;
            var local$$7037;
            var local$$7039;
            var local$$7041;
            var local$$7043;
            local$$7027 = local$$7006[lastIndexOf](local$$7048);
            if (local$$7027 < 0) {
              /** @type {number} */
              local$$7027 = 0;
            }
            /** @type {number} */
            local$$7029 = 0;
            for (; local$$7029 < local$$7027; ++local$$7029) {
              if (local$$7006[charCodeAt](local$$7029) >= 128) {
                local$$6712(not-basic);
              }
              local$$7009[push](local$$7006[charCodeAt](local$$7029));
            }
            local$$7031 = local$$7027 > 0 ? local$$7027 + 1 : 0;
            for (; local$$7031 < local$$7014;) {
              /** @type {number} */
              local$$7033 = local$$7019;
              /** @type {number} */
              local$$7035 = 1;
              /** @type {number} */
              local$$7037 = local$$6933;
              for (;; local$$7037 = local$$7037 + local$$6933) {
                if (local$$7031 >= local$$7014) {
                  local$$6712(invalid-input);
                }
                local$$7039 = local$$6900(local$$7006[charCodeAt](local$$7031++));
                if (local$$7039 >= local$$6933 || local$$7039 > local$$6964((local$$7130 - local$$7019) / local$$7035)) {
                  local$$6712(overflow);
                }
                /** @type {number} */
                local$$7019 = local$$7019 + local$$7039 * local$$7035;
                /** @type {number} */
                local$$7041 = local$$7037 <= local$$7025 ? local$$7148 : local$$7037 >= local$$7025 + local$$6980 ? local$$6980 : local$$7037 - local$$7025;
                if (local$$7039 < local$$7041) {
                  break;
                }
                /** @type {number} */
                local$$7043 = local$$6933 - local$$7041;
                if (local$$7035 > local$$6964(local$$7130 / local$$7043)) {
                  local$$6712(overflow);
                }
                /** @type {number} */
                local$$7035 = local$$7035 * local$$7043;
              }
              local$$7016 = local$$7009[length] + 1;
              local$$7025 = local$$6956(local$$7019 - local$$7033, local$$7016, local$$7033 == 0);
              if (local$$6964(local$$7019 / local$$7016) > local$$7130 - local$$7022) {
                local$$6712(overflow);
              }
              local$$7022 = local$$7022 + local$$6964(local$$7019 / local$$7016);
              /** @type {number} */
              local$$7019 = local$$7019 % local$$7016;
              local$$7009[splice](local$$7019++, 0, local$$7022);
            }
            return local$$6848(local$$7009);
          }
          /**
           * @param {!Array} local$$7227
           * @return {?}
           */
          function local$$7226(local$$7227) {
            var local$$7229;
            var local$$7231;
            var local$$7233;
            var local$$7235;
            var local$$7237;
            var local$$7239;
            var local$$7241;
            var local$$7243;
            var local$$7245;
            var local$$7247;
            var local$$7249;
            /** @type {!Array} */
            var local$$7252 = [];
            var local$$7254;
            var local$$7256;
            var local$$7258;
            var local$$7260;
            local$$7227 = local$$6764(local$$7227);
            local$$7254 = local$$7227[length];
            /** @type {number} */
            local$$7229 = local$$7021;
            /** @type {number} */
            local$$7231 = 0;
            /** @type {number} */
            local$$7237 = local$$7024;
            /** @type {number} */
            local$$7239 = 0;
            for (; local$$7239 < local$$7254; ++local$$7239) {
              local$$7249 = local$$7227[local$$7239];
              if (local$$7249 < 128) {
                local$$7252[push](local$$6863(local$$7249));
              }
            }
            local$$7233 = local$$7235 = local$$7252[length];
            if (local$$7235) {
              local$$7252[push](local$$7048);
            }
            for (; local$$7233 < local$$7254;) {
              /** @type {number} */
              local$$7241 = local$$7130;
              /** @type {number} */
              local$$7239 = 0;
              for (; local$$7239 < local$$7254; ++local$$7239) {
                local$$7249 = local$$7227[local$$7239];
                if (local$$7249 >= local$$7229 && local$$7249 < local$$7241) {
                  local$$7241 = local$$7249;
                }
              }
              local$$7256 = local$$7233 + 1;
              if (local$$7241 - local$$7229 > local$$6964((local$$7130 - local$$7231) / local$$7256)) {
                local$$6712(overflow);
              }
              /** @type {number} */
              local$$7231 = local$$7231 + (local$$7241 - local$$7229) * local$$7256;
              local$$7229 = local$$7241;
              /** @type {number} */
              local$$7239 = 0;
              for (; local$$7239 < local$$7254; ++local$$7239) {
                local$$7249 = local$$7227[local$$7239];
                if (local$$7249 < local$$7229 && ++local$$7231 > local$$7130) {
                  local$$6712(overflow);
                }
                if (local$$7249 == local$$7229) {
                  /** @type {number} */
                  local$$7243 = local$$7231;
                  /** @type {number} */
                  local$$7245 = local$$6933;
                  for (;; local$$7245 = local$$7245 + local$$6933) {
                    /** @type {number} */
                    local$$7247 = local$$7245 <= local$$7237 ? local$$7148 : local$$7245 >= local$$7237 + local$$6980 ? local$$6980 : local$$7245 - local$$7237;
                    if (local$$7243 < local$$7247) {
                      break;
                    }
                    /** @type {number} */
                    local$$7260 = local$$7243 - local$$7247;
                    /** @type {number} */
                    local$$7258 = local$$6933 - local$$7247;
                    local$$7252[push](local$$6863(local$$6937(local$$7247 + local$$7260 % local$$7258, 0)));
                    local$$7243 = local$$6964(local$$7260 / local$$7258);
                  }
                  local$$7252[push](local$$6863(local$$6937(local$$7243, 0)));
                  local$$7237 = local$$6956(local$$7231, local$$7256, local$$7233 == local$$7235);
                  /** @type {number} */
                  local$$7231 = 0;
                  ++local$$7233;
                }
              }
              ++local$$7231;
              ++local$$7229;
            }
            return local$$7252[join]();
          }
          /**
           * @param {?} local$$7464
           * @return {?}
           */
          function local$$7463(local$$7464) {
            return local$$6745(local$$7464, function(local$$7466) {
              return local$$7468[test](local$$7466) ? local$$7005(local$$7466[slice](4)[toLowerCase]()) : local$$7466;
            });
          }
          /**
           * @param {?} local$$7492
           * @return {?}
           */
          function local$$7491(local$$7492) {
            return local$$6745(local$$7492, function(local$$7494) {
              return local$$7496[test](local$$7494) ? xn-- + local$$7226(local$$7494) : local$$7494;
            });
          }
          var local$$7518 = typeof local$$6704 == object && local$$6704;
          var local$$7531 = typeof local$$6703 == object && local$$6703 && local$$6703[exports] == local$$7518 && local$$6703;
          var local$$7538 = typeof local$$6706 == object && local$$6706;
          if (local$$7538[global] === local$$7538 || local$$7538[window] === local$$7538) {
            local$$6710 = local$$7538;
          }
          var local$$7554;
          /** @type {number} */
          var local$$7130 = 2147483647;
          /** @type {number} */
          var local$$6933 = 36;
          /** @type {number} */
          var local$$7148 = 1;
          /** @type {number} */
          var local$$6980 = 26;
          /** @type {number} */
          var local$$6997 = 38;
          /** @type {number} */
          var local$$6965 = 700;
          /** @type {number} */
          var local$$7024 = 72;
          /** @type {number} */
          var local$$7021 = 128;
          var local$$7048 = -;
          /** @type {!RegExp} */
          var local$$7468 = /^xn--/;
          /** @type {!RegExp} */
          var local$$7496 = /[^ -~]/;
          /** @type {!RegExp} */
          var local$$6752 = /\x2E|\u3002|\uFF0E|\uFF61/g;
          var local$$6716 = {
            "overflow" : Overflow: input needs wider integers to process,
            "not-basic" : Illegal input >= 0x80 (not a basic code point),
            "invalid-input" : Invalid input
          };
          /** @type {number} */
          var local$$6979 = local$$6933 - local$$7148;
          var local$$6964 = Math[floor];
          var local$$6863 = String[fromCharCode];
          var local$$7603;
          local$$7554 = {
            "version" : 1.2.4,
            "ucs2" : {
              "decode" : local$$6764,
              "encode" : local$$6848
            },
            "decode" : local$$7005,
            "encode" : local$$7226,
            "toASCII" : local$$7491,
            "toUnicode" : local$$7463
          };
          if (typeof local$$6572 == function && typeof local$$6572[amd] == object && local$$6572[amd]) {
            local$$6572(punycode, function() {
              return local$$7554;
            });
          } else {
            if (local$$7518 && !local$$7518[nodeType]) {
              if (local$$7531) {
                local$$7531[exports] = local$$7554;
              } else {
                for (local$$7603 in local$$7554) {
                  if (local$$7554[hasOwnProperty](local$$7603)) {
                    local$$7518[local$$7603] = local$$7554[local$$7603];
                  }
                }
              }
            } else {
              local$$6710[punycode] = local$$7554;
            }
          }
        })(this);
      })[call](this, typeof global !== undefined ? global : typeof self !== undefined ? self : typeof window !== undefined ? window : {});
    }, {}],
    2 : [function(local$$7705, local$$7706, local$$7707) {
      /**
       * @param {?} local$$7710
       * @param {?} local$$7711
       * @param {?} local$$7712
       * @return {undefined}
       */
      function local$$7709(local$$7710, local$$7711, local$$7712) {
        if (local$$7710[defaultView] && (local$$7711 !== local$$7710[defaultView][pageXOffset] || local$$7712 !== local$$7710[defaultView][pageYOffset])) {
          local$$7710[defaultView][scrollTo](local$$7711, local$$7712);
        }
      }
      /**
       * @param {?} local$$7746
       * @param {?} local$$7747
       * @return {undefined}
       */
      function local$$7745(local$$7746, local$$7747) {
        try {
          if (local$$7747) {
            local$$7747[width] = local$$7746[width];
            local$$7747[height] = local$$7746[height];
            local$$7747[getContext](2d)[putImageData](local$$7746[getContext](2d)[getImageData](0, 0, local$$7746[width], local$$7746[height]), 0, 0);
          }
        } catch (local$$7799) {
          local$$7800(Unable to copy canvas content from, local$$7746, local$$7799);
        }
      }
      /**
       * @param {?} local$$7812
       * @param {boolean} local$$7813
       * @return {?}
       */
      function local$$7811(local$$7812, local$$7813) {
        var local$$7834 = local$$7812[nodeType] === 3 ? document[createTextNode](local$$7812[nodeValue]) : local$$7812[cloneNode](false);
        var local$$7839 = local$$7812[firstChild];
        for (; local$$7839;) {
          if (local$$7813 === true || local$$7839[nodeType] !== 1 || local$$7839[nodeName] !== SCRIPT) {
            local$$7834[appendChild](local$$7811(local$$7839, local$$7813));
          }
          local$$7839 = local$$7839[nextSibling];
        }
        if (local$$7812[nodeType] === 1) {
          local$$7834[_scrollTop] = local$$7812[scrollTop];
          local$$7834[_scrollLeft] = local$$7812[scrollLeft];
          if (local$$7812[nodeName] === CANVAS) {
            local$$7745(local$$7812, local$$7834);
          } else {
            if (local$$7812[nodeName] === TEXTAREA || local$$7812[nodeName] === SELECT) {
              local$$7834[value] = local$$7812[value];
            }
          }
        }
        return local$$7834;
      }
      /**
       * @param {?} local$$7937
       * @return {undefined}
       */
      function local$$7936(local$$7937) {
        if (local$$7937[nodeType] === 1) {
          local$$7937[scrollTop] = local$$7937[_scrollTop];
          local$$7937[scrollLeft] = local$$7937[_scrollLeft];
          var local$$7963 = local$$7937[firstChild];
          for (; local$$7963;) {
            local$$7936(local$$7963);
            local$$7963 = local$$7963[nextSibling];
          }
        }
      }
      var local$$7800 = local$$7705(./log);
      /**
       * @param {?} local$$7987
       * @param {?} local$$7988
       * @param {?} local$$7989
       * @param {?} local$$7990
       * @param {?} local$$7991
       * @param {?} local$$7992
       * @param {?} local$$7993
       * @return {?}
       */
      local$$7706[exports] = function(local$$7987, local$$7988, local$$7989, local$$7990, local$$7991, local$$7992, local$$7993) {
        var local$$8002 = local$$7811(local$$7987[documentElement], local$$7991[javascriptEnabled]);
        var local$$8010 = local$$7988[createElement](iframe);
        local$$8010[className] = html2canvas-container;
        local$$8010[style][visibility] = hidden;
        local$$8010[style][position] = fixed;
        local$$8010[style][left] = -10000px;
        local$$8010[style][top] = 0px;
        local$$8010[style][border] = 0;
        local$$8010[width] = local$$7989;
        local$$8010[height] = local$$7990;
        local$$8010[scrolling] = no;
        local$$7988[body][appendChild](local$$8010);
        return new Promise(function(local$$8095) {
          var local$$8103 = local$$8010[contentWindow][document];
          /** @type {function(): undefined} */
          local$$8010[contentWindow][onload] = local$$8010[onload] = function() {
            /** @type {number} */
            var local$$8134 = setInterval(function() {
              if (local$$8103[body][childNodes][length] > 0) {
                local$$7936(local$$8103[documentElement]);
                clearInterval(local$$8134);
                if (local$$7991[type] === view) {
                  local$$8010[contentWindow][scrollTo](local$$7992, local$$7993);
                  if (/(iPad|iPhone|iPod)/g[test](navigator[userAgent]) && (local$$8010[contentWindow][scrollY] !== local$$7993 || local$$8010[contentWindow][scrollX] !== local$$7992)) {
                    local$$8103[documentElement][style][top] = -local$$7993 + px;
                    local$$8103[documentElement][style][left] = -local$$7992 + px;
                    local$$8103[documentElement][style][position] = absolute;
                  }
                }
                local$$8095(local$$8010);
              }
            }, 50);
          };
          local$$8103[open]();
          local$$8103[write](<!DOCTYPE html><html></html>);
          local$$7709(local$$7987, local$$7992, local$$7993);
          local$$8103[replaceChild](local$$8103[adoptNode](local$$8002), local$$8103[documentElement]);
          local$$8103[close]();
        });
      };
    }, {
      "./log" : 13
    }],
    3 : [function(local$$8284, local$$8285, local$$8286) {
      /**
       * @param {?} local$$8289
       * @return {undefined}
       */
      function local$$8288(local$$8289) {
        /** @type {number} */
        this[r] = 0;
        /** @type {number} */
        this[g] = 0;
        /** @type {number} */
        this[b] = 0;
        /** @type {null} */
        this[a] = null;
        var local$$8344 = this[fromArray](local$$8289) || this[namedColor](local$$8289) || this[rgb](local$$8289) || this[rgba](local$$8289) || this[hex6](local$$8289) || this[hex3](local$$8289);
      }
      /**
       * @param {number} local$$8354
       * @return {?}
       */
      local$$8288[prototype][darken] = function(local$$8354) {
        /** @type {number} */
        var local$$8358 = 1 - local$$8354;
        return new local$$8288([Math[round](this[r] * local$$8358), Math[round](this[g] * local$$8358), Math[round](this[b] * local$$8358), this[a]]);
      };
      /**
       * @return {?}
       */
      local$$8288[prototype][isTransparent] = function() {
        return this[a] === 0;
      };
      /**
       * @return {?}
       */
      local$$8288[prototype][isBlack] = function() {
        return this[r] === 0 && this[g] === 0 && this[b] === 0;
      };
      /**
       * @param {!Array} local$$8446
       * @return {?}
       */
      local$$8288[prototype][fromArray] = function(local$$8446) {
        if (Array[isArray](local$$8446)) {
          this[r] = Math[min](local$$8446[0], 255);
          this[g] = Math[min](local$$8446[1], 255);
          this[b] = Math[min](local$$8446[2], 255);
          if (local$$8446[length] > 3) {
            this[a] = local$$8446[3];
          }
        }
        return Array[isArray](local$$8446);
      };
      /** @type {!RegExp} */
      var local$$8518 = /^#([a-f0-9]{3})$/i;
      /**
       * @param {?} local$$8526
       * @return {?}
       */
      local$$8288[prototype][hex3] = function(local$$8526) {
        /** @type {null} */
        var local$$8529 = null;
        if ((local$$8529 = local$$8526[match](local$$8518)) !== null) {
          /** @type {number} */
          this[r] = parseInt(local$$8529[1][0] + local$$8529[1][0], 16);
          /** @type {number} */
          this[g] = parseInt(local$$8529[1][1] + local$$8529[1][1], 16);
          /** @type {number} */
          this[b] = parseInt(local$$8529[1][2] + local$$8529[1][2], 16);
        }
        return local$$8529 !== null;
      };
      /** @type {!RegExp} */
      var local$$8599 = /^#([a-f0-9]{6})$/i;
      /**
       * @param {?} local$$8607
       * @return {?}
       */
      local$$8288[prototype][hex6] = function(local$$8607) {
        /** @type {null} */
        var local$$8610 = null;
        if ((local$$8610 = local$$8607[match](local$$8599)) !== null) {
          /** @type {number} */
          this[r] = parseInt(local$$8610[1][substring](0, 2), 16);
          /** @type {number} */
          this[g] = parseInt(local$$8610[1][substring](2, 4), 16);
          /** @type {number} */
          this[b] = parseInt(local$$8610[1][substring](4, 6), 16);
        }
        return local$$8610 !== null;
      };
      /** @type {!RegExp} */
      var local$$8676 = /^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$/;
      /**
       * @param {?} local$$8684
       * @return {?}
       */
      local$$8288[prototype][rgb] = function(local$$8684) {
        /** @type {null} */
        var local$$8687 = null;
        if ((local$$8687 = local$$8684[match](local$$8676)) !== null) {
          /** @type {number} */
          this[r] = Number(local$$8687[1]);
          /** @type {number} */
          this[g] = Number(local$$8687[2]);
          /** @type {number} */
          this[b] = Number(local$$8687[3]);
        }
        return local$$8687 !== null;
      };
      /** @type {!RegExp} */
      var local$$8733 = /^rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d?\.?\d+)\s*\)$/;
      /**
       * @param {?} local$$8741
       * @return {?}
       */
      local$$8288[prototype][rgba] = function(local$$8741) {
        /** @type {null} */
        var local$$8744 = null;
        if ((local$$8744 = local$$8741[match](local$$8733)) !== null) {
          /** @type {number} */
          this[r] = Number(local$$8744[1]);
          /** @type {number} */
          this[g] = Number(local$$8744[2]);
          /** @type {number} */
          this[b] = Number(local$$8744[3]);
          /** @type {number} */
          this[a] = Number(local$$8744[4]);
        }
        return local$$8744 !== null;
      };
      /**
       * @return {?}
       */
      local$$8288[prototype][toString] = function() {
        return this[a] !== null && this[a] !== 1 ? rgba( + [this[r], this[g], this[b], this[a]][join](,) + ) : rgb( + [this[r], this[g], this[b]][join](,) + );
      };
      /**
       * @param {?} local$$8872
       * @return {?}
       */
      local$$8288[prototype][namedColor] = function(local$$8872) {
        local$$8872 = local$$8872[toLowerCase]();
        var local$$8882 = local$$8880[local$$8872];
        if (local$$8882) {
          this[r] = local$$8882[0];
          this[g] = local$$8882[1];
          this[b] = local$$8882[2];
        } else {
          if (local$$8872 === transparent) {
            /** @type {number} */
            this[r] = this[g] = this[b] = this[a] = 0;
            return true;
          }
        }
        return !!local$$8882;
      };
      /** @type {boolean} */
      local$$8288[prototype][isColor] = true;
      var local$$8880 = {
        "aliceblue" : [240, 248, 255],
        "antiquewhite" : [250, 235, 215],
        "aqua" : [0, 255, 255],
        "aquamarine" : [127, 255, 212],
        "azure" : [240, 255, 255],
        "beige" : [245, 245, 220],
        "bisque" : [255, 228, 196],
        "black" : [0, 0, 0],
        "blanchedalmond" : [255, 235, 205],
        "blue" : [0, 0, 255],
        "blueviolet" : [138, 43, 226],
        "brown" : [165, 42, 42],
        "burlywood" : [222, 184, 135],
        "cadetblue" : [95, 158, 160],
        "chartreuse" : [127, 255, 0],
        "chocolate" : [210, 105, 30],
        "coral" : [255, 127, 80],
        "cornflowerblue" : [100, 149, 237],
        "cornsilk" : [255, 248, 220],
        "crimson" : [220, 20, 60],
        "cyan" : [0, 255, 255],
        "darkblue" : [0, 0, 139],
        "darkcyan" : [0, 139, 139],
        "darkgoldenrod" : [184, 134, 11],
        "darkgray" : [169, 169, 169],
        "darkgreen" : [0, 100, 0],
        "darkgrey" : [169, 169, 169],
        "darkkhaki" : [189, 183, 107],
        "darkmagenta" : [139, 0, 139],
        "darkolivegreen" : [85, 107, 47],
        "darkorange" : [255, 140, 0],
        "darkorchid" : [153, 50, 204],
        "darkred" : [139, 0, 0],
        "darksalmon" : [233, 150, 122],
        "darkseagreen" : [143, 188, 143],
        "darkslateblue" : [72, 61, 139],
        "darkslategray" : [47, 79, 79],
        "darkslategrey" : [47, 79, 79],
        "darkturquoise" : [0, 206, 209],
        "darkviolet" : [148, 0, 211],
        "deeppink" : [255, 20, 147],
        "deepskyblue" : [0, 191, 255],
        "dimgray" : [105, 105, 105],
        "dimgrey" : [105, 105, 105],
        "dodgerblue" : [30, 144, 255],
        "firebrick" : [178, 34, 34],
        "floralwhite" : [255, 250, 240],
        "forestgreen" : [34, 139, 34],
        "fuchsia" : [255, 0, 255],
        "gainsboro" : [220, 220, 220],
        "ghostwhite" : [248, 248, 255],
        "gold" : [255, 215, 0],
        "goldenrod" : [218, 165, 32],
        "gray" : [128, 128, 128],
        "green" : [0, 128, 0],
        "greenyellow" : [173, 255, 47],
        "grey" : [128, 128, 128],
        "honeydew" : [240, 255, 240],
        "hotpink" : [255, 105, 180],
        "indianred" : [205, 92, 92],
        "indigo" : [75, 0, 130],
        "ivory" : [255, 255, 240],
        "khaki" : [240, 230, 140],
        "lavender" : [230, 230, 250],
        "lavenderblush" : [255, 240, 245],
        "lawngreen" : [124, 252, 0],
        "lemonchiffon" : [255, 250, 205],
        "lightblue" : [173, 216, 230],
        "lightcoral" : [240, 128, 128],
        "lightcyan" : [224, 255, 255],
        "lightgoldenrodyellow" : [250, 250, 210],
        "lightgray" : [211, 211, 211],
        "lightgreen" : [144, 238, 144],
        "lightgrey" : [211, 211, 211],
        "lightpink" : [255, 182, 193],
        "lightsalmon" : [255, 160, 122],
        "lightseagreen" : [32, 178, 170],
        "lightskyblue" : [135, 206, 250],
        "lightslategray" : [119, 136, 153],
        "lightslategrey" : [119, 136, 153],
        "lightsteelblue" : [176, 196, 222],
        "lightyellow" : [255, 255, 224],
        "lime" : [0, 255, 0],
        "limegreen" : [50, 205, 50],
        "linen" : [250, 240, 230],
        "magenta" : [255, 0, 255],
        "maroon" : [128, 0, 0],
        "mediumaquamarine" : [102, 205, 170],
        "mediumblue" : [0, 0, 205],
        "mediumorchid" : [186, 85, 211],
        "mediumpurple" : [147, 112, 219],
        "mediumseagreen" : [60, 179, 113],
        "mediumslateblue" : [123, 104, 238],
        "mediumspringgreen" : [0, 250, 154],
        "mediumturquoise" : [72, 209, 204],
        "mediumvioletred" : [199, 21, 133],
        "midnightblue" : [25, 25, 112],
        "mintcream" : [245, 255, 250],
        "mistyrose" : [255, 228, 225],
        "moccasin" : [255, 228, 181],
        "navajowhite" : [255, 222, 173],
        "navy" : [0, 0, 128],
        "oldlace" : [253, 245, 230],
        "olive" : [128, 128, 0],
        "olivedrab" : [107, 142, 35],
        "orange" : [255, 165, 0],
        "orangered" : [255, 69, 0],
        "orchid" : [218, 112, 214],
        "palegoldenrod" : [238, 232, 170],
        "palegreen" : [152, 251, 152],
        "paleturquoise" : [175, 238, 238],
        "palevioletred" : [219, 112, 147],
        "papayawhip" : [255, 239, 213],
        "peachpuff" : [255, 218, 185],
        "peru" : [205, 133, 63],
        "pink" : [255, 192, 203],
        "plum" : [221, 160, 221],
        "powderblue" : [176, 224, 230],
        "purple" : [128, 0, 128],
        "rebeccapurple" : [102, 51, 153],
        "red" : [255, 0, 0],
        "rosybrown" : [188, 143, 143],
        "royalblue" : [65, 105, 225],
        "saddlebrown" : [139, 69, 19],
        "salmon" : [250, 128, 114],
        "sandybrown" : [244, 164, 96],
        "seagreen" : [46, 139, 87],
        "seashell" : [255, 245, 238],
        "sienna" : [160, 82, 45],
        "silver" : [192, 192, 192],
        "skyblue" : [135, 206, 235],
        "slateblue" : [106, 90, 205],
        "slategray" : [112, 128, 144],
        "slategrey" : [112, 128, 144],
        "snow" : [255, 250, 250],
        "springgreen" : [0, 255, 127],
        "steelblue" : [70, 130, 180],
        "tan" : [210, 180, 140],
        "teal" : [0, 128, 128],
        "thistle" : [216, 191, 216],
        "tomato" : [255, 99, 71],
        "turquoise" : [64, 224, 208],
        "violet" : [238, 130, 238],
        "wheat" : [245, 222, 179],
        "white" : [255, 255, 255],
        "whitesmoke" : [245, 245, 245],
        "yellow" : [255, 255, 0],
        "yellowgreen" : [154, 205, 50]
      };
      /** @type {function(?): undefined} */
      local$$8285[exports] = local$$8288;
    }, {}],
    4 : [function(local$$9699, local$$9700, local$$9701) {
      /**
       * @param {string} local$$9704
       * @param {!Object} local$$9705
       * @return {?}
       */
      function local$$9703(local$$9704, local$$9705) {
        /** @type {number} */
        var local$$9709 = local$$9707++;
        local$$9705 = local$$9705 || {};
        if (local$$9705[logging]) {
          /** @type {boolean} */
          local$$9718[options][logging] = true;
          local$$9718[options][start] = Date[now]();
        }
        local$$9705[async] = typeof local$$9705[async] === undefined ? true : local$$9705[async];
        local$$9705[allowTaint] = typeof local$$9705[allowTaint] === undefined ? false : local$$9705[allowTaint];
        local$$9705[removeContainer] = typeof local$$9705[removeContainer] === undefined ? true : local$$9705[removeContainer];
        local$$9705[javascriptEnabled] = typeof local$$9705[javascriptEnabled] === undefined ? false : local$$9705[javascriptEnabled];
        local$$9705[imageTimeout] = typeof local$$9705[imageTimeout] === undefined ? 1E4 : local$$9705[imageTimeout];
        local$$9705[renderer] = typeof local$$9705[renderer] === function ? local$$9705[renderer] : local$$9842;
        /** @type {boolean} */
        local$$9705[strict] = !!local$$9705[strict];
        if (typeof local$$9704 === string) {
          if (typeof local$$9705[proxy] !== string) {
            return Promise[reject](Proxy must be used when rendering url);
          }
          var local$$9889 = local$$9705[width] != null ? local$$9705[width] : window[innerWidth];
          var local$$9903 = local$$9705[height] != null ? local$$9705[height] : window[innerHeight];
          return local$$9905(local$$9906(local$$9704), local$$9705[proxy], document, local$$9889, local$$9903, local$$9705)[then](function(local$$9915) {
            return local$$9917(local$$9915[contentWindow][document][documentElement], local$$9915, local$$9705, local$$9889, local$$9903);
          });
        }
        var local$$9949 = (local$$9704 === undefined ? [document[documentElement]] : local$$9704[length] ? local$$9704 : [local$$9704])[0];
        local$$9949[setAttribute](local$$9954 + local$$9709, local$$9709);
        return local$$9958(local$$9949[ownerDocument], local$$9705, local$$9949[ownerDocument][defaultView][innerWidth], local$$9949[ownerDocument][defaultView][innerHeight], local$$9709)[then](function(local$$9984) {
          if (typeof local$$9705[onrendered] === function) {
            local$$9718(options.onrendered is deprecated, html2canvas returns a Promise containing the canvas);
            local$$9705[onrendered](local$$9984);
          }
          return local$$9984;
        });
      }
      /**
       * @param {?} local$$10012
       * @param {!Object} local$$10013
       * @param {(Element|!Function)} local$$10014
       * @param {(Element|!Function)} local$$10015
       * @param {number} local$$10016
       * @return {?}
       */
      function local$$9958(local$$10012, local$$10013, local$$10014, local$$10015, local$$10016) {
        return local$$10018(local$$10012, local$$10012, local$$10014, local$$10015, local$$10013, local$$10012[defaultView][pageXOffset], local$$10012[defaultView][pageYOffset])[then](function(local$$10035) {
          local$$9718(Document cloned);
          var local$$10042 = local$$9954 + local$$10016;
          var local$$10054 = [ + local$$10042 + =' + local$$10016 + '];
          local$$10012[querySelector](local$$10054)[removeAttribute](local$$10042);
          var local$$10068 = local$$10035[contentWindow];
          var local$$10077 = local$$10068[document][querySelector](local$$10054);
          var local$$10103 = typeof local$$10013[onclone] === function ? Promise[resolve](local$$10013[onclone](local$$10068[document])) : Promise[resolve](true);
          return local$$10103[then](function() {
            return local$$9917(local$$10077, local$$10035, local$$10013, local$$10014, local$$10015);
          });
        });
      }
      /**
       * @param {!Object} local$$10121
       * @param {?} local$$10122
       * @param {!Object} local$$10123
       * @param {(Element|!Function)} local$$10124
       * @param {(Element|!Function)} local$$10125
       * @return {?}
       */
      function local$$9917(local$$10121, local$$10122, local$$10123, local$$10124, local$$10125) {
        var local$$10130 = local$$10122[contentWindow];
        var local$$10137 = new local$$10132(local$$10130[document]);
        var local$$10141 = new local$$10139(local$$10123, local$$10137);
        var local$$10145 = local$$10143(local$$10121);
        var local$$10159 = local$$10123[type] === view ? local$$10124 : local$$10153(local$$10130[document]);
        var local$$10173 = local$$10123[type] === view ? local$$10125 : local$$10167(local$$10130[document]);
        var local$$10179 = new local$$10123[renderer](local$$10159, local$$10173, local$$10141, local$$10123, document);
        var local$$10183 = new local$$10181(local$$10121, local$$10179, local$$10137, local$$10141, local$$10123);
        return local$$10183[ready][then](function() {
          local$$9718(Finished rendering);
          var local$$10196;
          if (local$$10123[type] === view) {
            local$$10196 = local$$10204(local$$10179[canvas], {
              width : local$$10179[canvas][width],
              height : local$$10179[canvas][height],
              top : 0,
              left : 0,
              x : 0,
              y : 0
            });
          } else {
            if (local$$10121 === local$$10130[document][body] || local$$10121 === local$$10130[document][documentElement] || local$$10123[canvas] != null) {
              local$$10196 = local$$10179[canvas];
            } else {
              local$$10196 = local$$10204(local$$10179[canvas], {
                width : local$$10123[width] != null ? local$$10123[width] : local$$10145[width],
                height : local$$10123[height] != null ? local$$10123[height] : local$$10145[height],
                top : local$$10145[top],
                left : local$$10145[left],
                x : 0,
                y : 0
              });
            }
          }
          local$$10300(local$$10122, local$$10123);
          return local$$10196;
        });
      }
      /**
       * @param {?} local$$10310
       * @param {!Object} local$$10311
       * @return {undefined}
       */
      function local$$10300(local$$10310, local$$10311) {
        if (local$$10311[removeContainer]) {
          local$$10310[parentNode][removeChild](local$$10310);
          local$$9718(Cleaned up container);
        }
      }
      /**
       * @param {?} local$$10332
       * @param {?} local$$10333
       * @return {?}
       */
      function local$$10204(local$$10332, local$$10333) {
        var local$$10341 = document[createElement](canvas);
        var local$$10360 = Math[min](local$$10332[width] - 1, Math[max](0, local$$10333[left]));
        var local$$10381 = Math[min](local$$10332[width], Math[max](1, local$$10333[left] + local$$10333[width]));
        var local$$10400 = Math[min](local$$10332[height] - 1, Math[max](0, local$$10333[top]));
        var local$$10421 = Math[min](local$$10332[height], Math[max](1, local$$10333[top] + local$$10333[height]));
        local$$10341[width] = local$$10333[width];
        local$$10341[height] = local$$10333[height];
        /** @type {number} */
        var local$$10440 = local$$10381 - local$$10360;
        /** @type {number} */
        var local$$10443 = local$$10421 - local$$10400;
        local$$9718(Cropping canvas at:, left:, local$$10333[left], top:, local$$10333[top], width:, local$$10440, height:, local$$10443);
        local$$9718(Resulting crop with width, local$$10333[width], and height, local$$10333[height], with x, local$$10360, and y, local$$10400);
        local$$10341[getContext](2d)[drawImage](local$$10332, local$$10360, local$$10400, local$$10440, local$$10443, local$$10333[x], local$$10333[y], local$$10440, local$$10443);
        return local$$10341;
      }
      /**
       * @param {?} local$$10499
       * @return {?}
       */
      function local$$10153(local$$10499) {
        return Math[max](Math[max](local$$10499[body][scrollWidth], local$$10499[documentElement][scrollWidth]), Math[max](local$$10499[body][offsetWidth], local$$10499[documentElement][offsetWidth]), Math[max](local$$10499[body][clientWidth], local$$10499[documentElement][clientWidth]));
      }
      /**
       * @param {?} local$$10556
       * @return {?}
       */
      function local$$10167(local$$10556) {
        return Math[max](Math[max](local$$10556[body][scrollHeight], local$$10556[documentElement][scrollHeight]), Math[max](local$$10556[body][offsetHeight], local$$10556[documentElement][offsetHeight]), Math[max](local$$10556[body][clientHeight], local$$10556[documentElement][clientHeight]));
      }
      /**
       * @param {string} local$$10613
       * @return {?}
       */
      function local$$9906(local$$10613) {
        var local$$10621 = document[createElement](a);
        /** @type {string} */
        local$$10621[href] = local$$10613;
        local$$10621[href] = local$$10621[href];
        return local$$10621;
      }
      var local$$10132 = local$$9699(./support);
      var local$$9842 = local$$9699(./renderers/canvas);
      var local$$10139 = local$$9699(./imageloader);
      var local$$10181 = local$$9699(./nodeparser);
      var local$$10658 = local$$9699(./nodecontainer);
      var local$$9718 = local$$9699(./log);
      var local$$10667 = local$$9699(./utils);
      var local$$10018 = local$$9699(./clone);
      var local$$9905 = local$$9699(./proxy)[loadUrlDocument];
      var local$$10143 = local$$10667[getBounds];
      var local$$9954 = data-html2canvas-node;
      /** @type {number} */
      var local$$9707 = 0;
      local$$9703[CanvasRenderer] = local$$9842;
      local$$9703[NodeContainer] = local$$10658;
      local$$9703[log] = local$$9718;
      local$$9703[utils] = local$$10667;
      /** @type {!Function} */
      var local$$10746 = typeof document === undefined || typeof Object[create] !== function || typeof document[createElement](canvas)[getContext] !== function ? function() {
        return Promise[reject](No canvas support);
      } : local$$9703;
      /** @type {!Function} */
      local$$9700[exports] = local$$10746;
      if (typeof local$$6572 === function && local$$6572[amd]) {
        local$$6572(html2canvas, [], function() {
          return local$$10746;
        });
      }
    }, {
      "./clone" : 2,
      "./imageloader" : 11,
      "./log" : 13,
      "./nodecontainer" : 14,
      "./nodeparser" : 15,
      "./proxy" : 16,
      "./renderers/canvas" : 20,
      "./support" : 22,
      "./utils" : 26
    }],
    5 : [function(local$$10787, local$$10788, local$$10789) {
      /**
       * @param {?} local$$10792
       * @return {undefined}
       */
      function local$$10791(local$$10792) {
        this[src] = local$$10792;
        local$$10799(DummyImageContainer for, local$$10792);
        if (!this[promise] || !this[image]) {
          local$$10799(Initiating DummyImageContainer);
          /** @type {!Image} */
          local$$10791[prototype][image] = new Image;
          var local$$10830 = this[image];
          /** @type {!Promise} */
          local$$10791[prototype][promise] = new Promise(function(local$$10838, local$$10839) {
            local$$10830[onload] = local$$10838;
            local$$10830[onerror] = local$$10839;
            local$$10830[src] = local$$10854();
            if (local$$10830[complete] === true) {
              local$$10838(local$$10830);
            }
          });
        }
      }
      var local$$10799 = local$$10787(./log);
      var local$$10854 = local$$10787(./utils)[smallImage];
      /** @type {function(?): undefined} */
      local$$10788[exports] = local$$10791;
    }, {
      "./log" : 13,
      "./utils" : 26
    }],
    6 : [function(local$$10899, local$$10900, local$$10901) {
      /**
       * @param {?} local$$10904
       * @param {?} local$$10905
       * @return {undefined}
       */
      function local$$10903(local$$10904, local$$10905) {
        var local$$10913 = document[createElement](div);
        var local$$10921 = document[createElement](img);
        var local$$10929 = document[createElement](span);
        var local$$10933 = Hidden Text;
        var local$$10935;
        var local$$10937;
        local$$10913[style][visibility] = hidden;
        local$$10913[style][fontFamily] = local$$10904;
        local$$10913[style][fontSize] = local$$10905;
        /** @type {number} */
        local$$10913[style][margin] = 0;
        /** @type {number} */
        local$$10913[style][padding] = 0;
        document[body][appendChild](local$$10913);
        local$$10921[src] = local$$10994();
        /** @type {number} */
        local$$10921[width] = 1;
        /** @type {number} */
        local$$10921[height] = 1;
        /** @type {number} */
        local$$10921[style][margin] = 0;
        /** @type {number} */
        local$$10921[style][padding] = 0;
        local$$10921[style][verticalAlign] = baseline;
        local$$10929[style][fontFamily] = local$$10904;
        local$$10929[style][fontSize] = local$$10905;
        /** @type {number} */
        local$$10929[style][margin] = 0;
        /** @type {number} */
        local$$10929[style][padding] = 0;
        local$$10929[appendChild](document[createTextNode](local$$10933));
        local$$10913[appendChild](local$$10929);
        local$$10913[appendChild](local$$10921);
        /** @type {number} */
        local$$10935 = local$$10921[offsetTop] - local$$10929[offsetTop] + 1;
        local$$10913[removeChild](local$$10929);
        local$$10913[appendChild](document[createTextNode](local$$10933));
        local$$10913[style][lineHeight] = normal;
        local$$10921[style][verticalAlign] = super;
        /** @type {number} */
        local$$10937 = local$$10921[offsetTop] - local$$10913[offsetTop] + 1;
        document[body][removeChild](local$$10913);
        /** @type {number} */
        this[baseline] = local$$10935;
        /** @type {number} */
        this[lineWidth] = 1;
        /** @type {number} */
        this[middle] = local$$10937;
      }
      var local$$10994 = local$$10899(./utils)[smallImage];
      /** @type {function(?, ?): undefined} */
      local$$10900[exports] = local$$10903;
    }, {
      "./utils" : 26
    }],
    7 : [function(local$$11191, local$$11192, local$$11193) {
      /**
       * @return {undefined}
       */
      function local$$11195() {
        this[data] = {};
      }
      var local$$11208 = local$$11191(./font);
      /**
       * @param {?} local$$11216
       * @param {?} local$$11217
       * @return {?}
       */
      local$$11195[prototype][getMetrics] = function(local$$11216, local$$11217) {
        if (this[data][local$$11216 + - + local$$11217] === undefined) {
          this[data][local$$11216 + - + local$$11217] = new local$$11208(local$$11216, local$$11217);
        }
        return this[data][local$$11216 + - + local$$11217];
      };
      /** @type {function(): undefined} */
      local$$11192[exports] = local$$11195;
    }, {
      "./font" : 6
    }],
    8 : [function(local$$11266, local$$11267, local$$11268) {
      /**
       * @param {?} local$$11271
       * @param {?} local$$11272
       * @param {?} local$$11273
       * @return {undefined}
       */
      function local$$11270(local$$11271, local$$11272, local$$11273) {
        /** @type {null} */
        this[image] = null;
        this[src] = local$$11271;
        var local$$11286 = this;
        var local$$11290 = local$$11288(local$$11271);
        this[promise] = (!local$$11272 ? this[proxyLoad](local$$11273[proxy], local$$11290, local$$11273) : new Promise(function(local$$11303) {
          if (local$$11271[contentWindow][document][URL] === about:blank || local$$11271[contentWindow][document][documentElement] == null) {
            /** @type {function(): undefined} */
            local$$11271[contentWindow][onload] = local$$11271[onload] = function() {
              local$$11303(local$$11271);
            };
          } else {
            local$$11303(local$$11271);
          }
        }))[then](function(local$$11358) {
          var local$$11363 = local$$11266(./core);
          return local$$11363(local$$11358[contentWindow][document][documentElement], {
            type : view,
            width : local$$11358[width],
            height : local$$11358[height],
            proxy : local$$11273[proxy],
            javascriptEnabled : local$$11273[javascriptEnabled],
            removeContainer : local$$11273[removeContainer],
            allowTaint : local$$11273[allowTaint],
            imageTimeout : local$$11273[imageTimeout] / 2
          });
        })[then](function(local$$11408) {
          return local$$11286[image] = local$$11408;
        });
      }
      var local$$11425 = local$$11266(./utils);
      var local$$11288 = local$$11425[getBounds];
      var local$$11437 = local$$11266(./proxy)[loadUrlDocument];
      /**
       * @param {?} local$$11445
       * @param {?} local$$11446
       * @param {?} local$$11447
       * @return {?}
       */
      local$$11270[prototype][proxyLoad] = function(local$$11445, local$$11446, local$$11447) {
        var local$$11452 = this[src];
        return local$$11437(local$$11452[src], local$$11445, local$$11452[ownerDocument], local$$11446[width], local$$11446[height], local$$11447);
      };
      /** @type {function(?, ?, ?): undefined} */
      local$$11267[exports] = local$$11270;
    }, {
      "./core" : 4,
      "./proxy" : 16,
      "./utils" : 26
    }],
    9 : [function(local$$11484, local$$11485, local$$11486) {
      /**
       * @param {?} local$$11489
       * @return {undefined}
       */
      function local$$11488(local$$11489) {
        this[src] = local$$11489[value];
        /** @type {!Array} */
        this[colorStops] = [];
        /** @type {null} */
        this[type] = null;
        /** @type {number} */
        this[x0] = .5;
        /** @type {number} */
        this[y0] = .5;
        /** @type {number} */
        this[x1] = .5;
        /** @type {number} */
        this[y1] = .5;
        this[promise] = Promise[resolve](true);
      }
      local$$11488[TYPES] = {
        LINEAR : 1,
        RADIAL : 2
      };
      /** @type {!RegExp} */
      local$$11488[REGEXP_COLORSTOP] = /^\s*(rgba?\(\s*\d{1,3},\s*\d{1,3},\s*\d{1,3}(?:,\s*[0-9\.]+)?\s*\)|[a-z]{3,20}|#[a-f0-9]{3,6})(?:\s+(\d{1,3}(?:\.\d+)?)(%|px)?)?(?:\s|$)/i;
      /** @type {function(?): undefined} */
      local$$11485[exports] = local$$11488;
    }, {}],
    10 : [function(local$$11572, local$$11573, local$$11574) {
      /**
       * @param {?} local$$11577
       * @param {?} local$$11578
       * @return {undefined}
       */
      function local$$11576(local$$11577, local$$11578) {
        this[src] = local$$11577;
        /** @type {!Image} */
        this[image] = new Image;
        var local$$11591 = this;
        /** @type {null} */
        this[tainted] = null;
        /** @type {!Promise} */
        this[promise] = new Promise(function(local$$11602, local$$11603) {
          local$$11591[image][onload] = local$$11602;
          local$$11591[image][onerror] = local$$11603;
          if (local$$11578) {
            local$$11591[image][crossOrigin] = anonymous;
          }
          local$$11591[image][src] = local$$11577;
          if (local$$11591[image][complete] === true) {
            local$$11602(local$$11591[image]);
          }
        });
      }
      /** @type {function(?, ?): undefined} */
      local$$11573[exports] = local$$11576;
    }, {}],
    11 : [function(local$$11674, local$$11675, local$$11676) {
      /**
       * @param {?} local$$11679
       * @param {?} local$$11680
       * @return {undefined}
       */
      function local$$11678(local$$11679, local$$11680) {
        /** @type {null} */
        this[link] = null;
        this[options] = local$$11679;
        this[support] = local$$11680;
        this[origin] = this[getOrigin](window[location][href]);
      }
      var local$$11718 = local$$11674(./log);
      var local$$11723 = local$$11674(./imagecontainer);
      var local$$11728 = local$$11674(./dummyimagecontainer);
      var local$$11733 = local$$11674(./proxyimagecontainer);
      var local$$11738 = local$$11674(./framecontainer);
      var local$$11743 = local$$11674(./svgcontainer);
      var local$$11748 = local$$11674(./svgnodecontainer);
      var local$$11753 = local$$11674(./lineargradientcontainer);
      var local$$11758 = local$$11674(./webkitgradientcontainer);
      var local$$11766 = local$$11674(./utils)[bind];
      /**
       * @param {?} local$$11774
       * @return {?}
       */
      local$$11678[prototype][findImages] = function(local$$11774) {
        /** @type {!Array} */
        var local$$11777 = [];
        local$$11774[reduce](function(local$$11782, local$$11783) {
          switch(local$$11783[node][nodeName]) {
            case IMG:
              return local$$11782[concat]([{
                args : [local$$11783[node][src]],
                method : url
              }]);
            case svg:
            case IFRAME:
              return local$$11782[concat]([{
                args : [local$$11783[node]],
                method : local$$11783[node][nodeName]
              }]);
          }
          return local$$11782;
        }, [])[forEach](this[addImage](local$$11777, this[loadImage]), this);
        return local$$11777;
      };
      /**
       * @param {?} local$$11867
       * @param {?} local$$11868
       * @return {?}
       */
      local$$11678[prototype][findBackgroundImage] = function(local$$11867, local$$11868) {
        local$$11868[parseBackgroundImages]()[filter](this[hasImageBackground])[forEach](this[addImage](local$$11867, this[loadImage]), this);
        return local$$11867;
      };
      /**
       * @param {?} local$$11904
       * @param {?} local$$11905
       * @return {?}
       */
      local$$11678[prototype][addImage] = function(local$$11904, local$$11905) {
        return function(local$$11907) {
          local$$11907[args][forEach](function(local$$11915) {
            if (!this[imageExists](local$$11904, local$$11915)) {
              local$$11904[splice](0, 0, local$$11905[call](this, local$$11907));
              local$$11718(Added image # + local$$11904[length], typeof local$$11915 === string ? local$$11915[substring](0, 100) : local$$11915);
            }
          }, this);
        };
      };
      /**
       * @param {?} local$$11971
       * @return {?}
       */
      local$$11678[prototype][hasImageBackground] = function(local$$11971) {
        return local$$11971[method] !== none;
      };
      /**
       * @param {?} local$$11990
       * @return {?}
       */
      local$$11678[prototype][loadImage] = function(local$$11990) {
        if (local$$11990[method] === url) {
          var local$$12003 = local$$11990[args][0];
          if (this[isSVG](local$$12003) && !this[support][svg] && !this[options][allowTaint]) {
            return new local$$11743(local$$12003);
          } else {
            if (local$$12003[match](/data:image\/.*;base64,/i)) {
              return new local$$11723(local$$12003[replace](/url\(['"]{0,}|['"]{0,}\)$/ig, ), false);
            } else {
              if (this[isSameOrigin](local$$12003) || this[options][allowTaint] === true || this[isSVG](local$$12003)) {
                return new local$$11723(local$$12003, false);
              } else {
                if (this[support][cors] && !this[options][allowTaint] && this[options][useCORS]) {
                  return new local$$11723(local$$12003, true);
                } else {
                  if (this[options][proxy]) {
                    return new local$$11733(local$$12003, this[options][proxy]);
                  } else {
                    return new local$$11728(local$$12003);
                  }
                }
              }
            }
          }
        } else {
          if (local$$11990[method] === linear-gradient) {
            return new local$$11753(local$$11990);
          } else {
            if (local$$11990[method] === gradient) {
              return new local$$11758(local$$11990);
            } else {
              if (local$$11990[method] === svg) {
                return new local$$11748(local$$11990[args][0], this[support][svg]);
              } else {
                if (local$$11990[method] === IFRAME) {
                  return new local$$11738(local$$11990[args][0], this[isSameOrigin](local$$11990[args][0][src]), this[options]);
                } else {
                  return new local$$11728(local$$11990);
                }
              }
            }
          }
        }
      };
      /**
       * @param {?} local$$12211
       * @return {?}
       */
      local$$11678[prototype][isSVG] = function(local$$12211) {
        return local$$12211[substring](local$$12211[length] - 3)[toLowerCase]() === svg || local$$11743[prototype][isInline](local$$12211);
      };
      /**
       * @param {?} local$$12248
       * @param {?} local$$12249
       * @return {?}
       */
      local$$11678[prototype][imageExists] = function(local$$12248, local$$12249) {
        return local$$12248[some](function(local$$12254) {
          return local$$12254[src] === local$$12249;
        });
      };
      /**
       * @param {?} local$$12275
       * @return {?}
       */
      local$$11678[prototype][isSameOrigin] = function(local$$12275) {
        return this[getOrigin](local$$12275) === this[origin];
      };
      /**
       * @param {?} local$$12296
       * @return {?}
       */
      local$$11678[prototype][getOrigin] = function(local$$12296) {
        var local$$12312 = this[link] || (this[link] = document[createElement](a));
        local$$12312[href] = local$$12296;
        local$$12312[href] = local$$12312[href];
        return local$$12312[protocol] + local$$12312[hostname] + local$$12312[port];
      };
      /**
       * @param {?} local$$12349
       * @return {?}
       */
      local$$11678[prototype][getPromise] = function(local$$12349) {
        return this[timeout](local$$12349, this[options][imageTimeout])[catch](function() {
          var local$$12369 = new local$$11728(local$$12349[src]);
          return local$$12369[promise][then](function(local$$12377) {
            local$$12349[image] = local$$12377;
          });
        });
      };
      /**
       * @param {?} local$$12402
       * @return {?}
       */
      local$$11678[prototype][get] = function(local$$12402) {
        /** @type {null} */
        var local$$12405 = null;
        return this[images][some](function(local$$12413) {
          return (local$$12405 = local$$12413)[src] === local$$12402;
        }) ? local$$12405 : null;
      };
      /**
       * @param {?} local$$12437
       * @return {?}
       */
      local$$11678[prototype][fetch] = function(local$$12437) {
        this[images] = local$$12437[reduce](local$$11766(this[findBackgroundImage], this), this[findImages](local$$12437));
        this[images][forEach](function(local$$12462, local$$12463) {
          local$$12462[promise][then](function() {
            local$$11718(Succesfully loaded image # + (local$$12463 + 1), local$$12462);
          }, function(local$$12481) {
            local$$11718(Failed loading image # + (local$$12463 + 1), local$$12462, local$$12481);
          });
        });
        this[ready] = Promise[all](this[images][map](this[getPromise], this));
        local$$11718(Finished searching images);
        return this;
      };
      /**
       * @param {?} local$$12532
       * @param {?} local$$12533
       * @return {?}
       */
      local$$11678[prototype][timeout] = function(local$$12532, local$$12533) {
        var local$$12535;
        var local$$12576 = Promise[race]([local$$12532[promise], new Promise(function(local$$12543, local$$12544) {
          /** @type {number} */
          local$$12535 = setTimeout(function() {
            local$$11718(Timed out loading image, local$$12532);
            local$$12544(local$$12532);
          }, local$$12533);
        })])[then](function(local$$12567) {
          clearTimeout(local$$12535);
          return local$$12567;
        });
        local$$12576[catch](function() {
          clearTimeout(local$$12535);
        });
        return local$$12576;
      };
      /** @type {function(?, ?): undefined} */
      local$$11675[exports] = local$$11678;
    }, {
      "./dummyimagecontainer" : 5,
      "./framecontainer" : 8,
      "./imagecontainer" : 10,
      "./lineargradientcontainer" : 12,
      "./log" : 13,
      "./proxyimagecontainer" : 17,
      "./svgcontainer" : 23,
      "./svgnodecontainer" : 24,
      "./utils" : 26,
      "./webkitgradientcontainer" : 27
    }],
    12 : [function(local$$12613, local$$12614, local$$12615) {
      /**
       * @param {?} local$$12618
       * @return {undefined}
       */
      function local$$12617(local$$12618) {
        local$$12620[apply](this, arguments);
        this[type] = local$$12620[TYPES][LINEAR];
        var local$$12664 = local$$12617[REGEXP_DIRECTION][test](local$$12618[args][0]) || !local$$12620[REGEXP_COLORSTOP][test](local$$12618[args][0]);
        if (local$$12664) {
          local$$12618[args][0][split](/\s+/)[reverse]()[forEach](function(local$$12684, local$$12685) {
            switch(local$$12684) {
              case left:
                /** @type {number} */
                this[x0] = 0;
                /** @type {number} */
                this[x1] = 1;
                break;
              case top:
                /** @type {number} */
                this[y0] = 0;
                /** @type {number} */
                this[y1] = 1;
                break;
              case right:
                /** @type {number} */
                this[x0] = 1;
                /** @type {number} */
                this[x1] = 0;
                break;
              case bottom:
                /** @type {number} */
                this[y0] = 1;
                /** @type {number} */
                this[y1] = 0;
                break;
              case to:
                var local$$12760 = this[y0];
                var local$$12765 = this[x0];
                this[y0] = this[y1];
                this[x0] = this[x1];
                this[x1] = local$$12765;
                this[y1] = local$$12760;
                break;
              case center:
                break;
              default:
                /** @type {number} */
                var local$$12806 = parseFloat(local$$12684, 10) * .01;
                if (isNaN(local$$12806)) {
                  break;
                }
                if (local$$12685 === 0) {
                  /** @type {number} */
                  this[y0] = local$$12806;
                  /** @type {number} */
                  this[y1] = 1 - this[y0];
                } else {
                  /** @type {number} */
                  this[x0] = local$$12806;
                  /** @type {number} */
                  this[x1] = 1 - this[x0];
                }
                break;
            }
          }, this);
        } else {
          /** @type {number} */
          this[y0] = 0;
          /** @type {number} */
          this[y1] = 1;
        }
        this[colorStops] = local$$12618[args][slice](local$$12664 ? 1 : 0)[map](function(local$$12890) {
          var local$$12897 = local$$12890[match](local$$12620.REGEXP_COLORSTOP);
          /** @type {number} */
          var local$$12902 = +local$$12897[2];
          var local$$12911 = local$$12902 === 0 ? % : local$$12897[3];
          return {
            color : new local$$12913(local$$12897[1]),
            stop : local$$12911 === % ? local$$12902 / 100 : null
          };
        });
        if (this[colorStops][0][stop] === null) {
          /** @type {number} */
          this[colorStops][0][stop] = 0;
        }
        if (this[colorStops][this[colorStops][length] - 1][stop] === null) {
          /** @type {number} */
          this[colorStops][this[colorStops][length] - 1][stop] = 1;
        }
        this[colorStops][forEach](function(local$$12999, local$$13000) {
          if (local$$12999[stop] === null) {
            this[colorStops][slice](local$$13000)[some](function(local$$13017, local$$13018) {
              if (local$$13017[stop] !== null) {
                local$$12999[stop] = (local$$13017[stop] - this[colorStops][local$$13000 - 1][stop]) / (local$$13018 + 1) + this[colorStops][local$$13000 - 1][stop];
                return true;
              } else {
                return false;
              }
            }, this);
          }
        }, this);
      }
      var local$$12620 = local$$12613(./gradientcontainer);
      var local$$12913 = local$$12613(./color);
      local$$12617[prototype] = Object[create](local$$12620[prototype]);
      /** @type {!RegExp} */
      local$$12617[REGEXP_DIRECTION] = /^\s*(?:to|left|right|top|bottom|center|\d{1,3}(?:\.\d+)?%?)(?:\s|$)/i;
      /** @type {function(?): undefined} */
      local$$12614[exports] = local$$12617;
    }, {
      "./color" : 3,
      "./gradientcontainer" : 9
    }],
    13 : [function(local$$13114, local$$13115, local$$13116) {
      /**
       * @return {undefined}
       */
      var local$$13119 = function() {
        if (local$$13119[options][logging] && window[console] && window[console][log]) {
          Function[prototype][bind][call](window[console][log], window[console])[apply](window[console], [Date[now]() - local$$13119[options][start] + ms, html2canvas:][concat]([][slice][call](arguments, 0)));
        }
      };
      local$$13119[options] = {
        logging : false
      };
      /** @type {function(): undefined} */
      local$$13115[exports] = local$$13119;
    }, {}],
    14 : [function(local$$13217, local$$13218, local$$13219) {
      /**
       * @param {?} local$$13222
       * @param {?} local$$13223
       * @return {undefined}
       */
      function local$$13221(local$$13222, local$$13223) {
        this[node] = local$$13222;
        this[parent] = local$$13223;
        /** @type {null} */
        this[stack] = null;
        /** @type {null} */
        this[bounds] = null;
        /** @type {null} */
        this[borders] = null;
        /** @type {!Array} */
        this[clip] = [];
        /** @type {!Array} */
        this[backgroundClip] = [];
        /** @type {null} */
        this[offsetBounds] = null;
        /** @type {null} */
        this[visible] = null;
        /** @type {null} */
        this[computedStyles] = null;
        this[colors] = {};
        this[styles] = {};
        /** @type {null} */
        this[backgroundImages] = null;
        /** @type {null} */
        this[transformData] = null;
        /** @type {null} */
        this[transformMatrix] = null;
        /** @type {boolean} */
        this[isPseudoElement] = false;
        /** @type {null} */
        this[opacity] = null;
      }
      /**
       * @param {?} local$$13328
       * @return {?}
       */
      function local$$13327(local$$13328) {
        var local$$13339 = local$$13328[options][local$$13328[selectedIndex] || 0];
        return local$$13339 ? local$$13339[text] ||  : ;
      }
      /**
       * @param {!Array} local$$13354
       * @return {?}
       */
      function local$$13353(local$$13354) {
        if (local$$13354 && local$$13354[1] === matrix) {
          return local$$13354[2][split](,)[map](function(local$$13373) {
            return parseFloat(local$$13373[trim]());
          });
        } else {
          if (local$$13354 && local$$13354[1] === matrix3d) {
            var local$$13414 = local$$13354[2][split](,)[map](function(local$$13403) {
              return parseFloat(local$$13403[trim]());
            });
            return [local$$13414[0], local$$13414[1], local$$13414[4], local$$13414[5], local$$13414[12], local$$13414[13]];
          }
        }
      }
      /**
       * @param {string} local$$13437
       * @return {?}
       */
      function local$$13436(local$$13437) {
        return local$$13437.toString()[indexOf](%) !== -1;
      }
      /**
       * @param {?} local$$13453
       * @return {?}
       */
      function local$$13452(local$$13453) {
        return local$$13453[replace](px, );
      }
      /**
       * @param {?} local$$13467
       * @return {?}
       */
      function local$$13466(local$$13467) {
        return parseFloat(local$$13467);
      }
      var local$$13476 = local$$13217(./color);
      var local$$13481 = local$$13217(./utils);
      var local$$13486 = local$$13481[getBounds];
      var local$$13491 = local$$13481[parseBackgrounds];
      var local$$13496 = local$$13481[offsetBounds];
      /**
       * @param {?} local$$13504
       * @return {undefined}
       */
      local$$13221[prototype][cloneTo] = function(local$$13504) {
        local$$13504[visible] = this[visible];
        local$$13504[borders] = this[borders];
        local$$13504[bounds] = this[bounds];
        local$$13504[clip] = this[clip];
        local$$13504[backgroundClip] = this[backgroundClip];
        local$$13504[computedStyles] = this[computedStyles];
        local$$13504[styles] = this[styles];
        local$$13504[backgroundImages] = this[backgroundImages];
        local$$13504[opacity] = this[opacity];
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][getOpacity] = function() {
        return this[opacity] === null ? this[opacity] = this[cssFloat](opacity) : this[opacity];
      };
      /**
       * @param {?} local$$13619
       * @return {undefined}
       */
      local$$13221[prototype][assignStack] = function(local$$13619) {
        this[stack] = local$$13619;
        local$$13619[children][push](this);
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][isElementVisible] = function() {
        return this[node][nodeType] === Node[TEXT_NODE] ? this[parent][visible] : this[css](display) !== none && this[css](visibility) !== hidden && !this[node][hasAttribute](data-html2canvas-ignore) && (this[node][nodeName] !== INPUT || this[node][getAttribute](type) !== hidden);
      };
      /**
       * @param {?} local$$13727
       * @return {?}
       */
      local$$13221[prototype][css] = function(local$$13727) {
        if (!this[computedStyles]) {
          this[computedStyles] = this[isPseudoElement] ? this[parent][computedStyle](this[before] ? :before : :after) : this[computedStyle](null);
        }
        return this[styles][local$$13727] || (this[styles][local$$13727] = this[computedStyles][local$$13727]);
      };
      /**
       * @param {?} local$$13790
       * @return {?}
       */
      local$$13221[prototype][prefixedCss] = function(local$$13790) {
        /** @type {!Array} */
        var local$$13801 = [webkit, moz, ms, o];
        var local$$13807 = this[css](local$$13790);
        if (local$$13807 === undefined) {
          local$$13801[some](function(local$$13813) {
            local$$13807 = this[css](local$$13813 + local$$13790[substr](0, 1)[toUpperCase]() + local$$13790[substr](1));
            return local$$13807 !== undefined;
          }, this);
        }
        return local$$13807 === undefined ? null : local$$13807;
      };
      /**
       * @param {?} local$$13861
       * @return {?}
       */
      local$$13221[prototype][computedStyle] = function(local$$13861) {
        return this[node][ownerDocument][defaultView][getComputedStyle](this[node], local$$13861);
      };
      /**
       * @param {?} local$$13890
       * @return {?}
       */
      local$$13221[prototype][cssInt] = function(local$$13890) {
        /** @type {number} */
        var local$$13898 = parseInt(this[css](local$$13890), 10);
        return isNaN(local$$13898) ? 0 : local$$13898;
      };
      /**
       * @param {?} local$$13914
       * @return {?}
       */
      local$$13221[prototype][color] = function(local$$13914) {
        return this[colors][local$$13914] || (this[colors][local$$13914] = new local$$13476(this[css](local$$13914)));
      };
      /**
       * @param {?} local$$13942
       * @return {?}
       */
      local$$13221[prototype][cssFloat] = function(local$$13942) {
        /** @type {number} */
        var local$$13949 = parseFloat(this[css](local$$13942));
        return isNaN(local$$13949) ? 0 : local$$13949;
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][fontWeight] = function() {
        var local$$13972 = this[css](fontWeight);
        switch(parseInt(local$$13972, 10)) {
          case 401:
            local$$13972 = bold;
            break;
          case 400:
            local$$13972 = normal;
            break;
        }
        return local$$13972;
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][parseClip] = function() {
        var local$$14017 = this[css](clip)[match](this.CLIP);
        if (local$$14017) {
          return {
            top : parseInt(local$$14017[1], 10),
            right : parseInt(local$$14017[2], 10),
            bottom : parseInt(local$$14017[3], 10),
            left : parseInt(local$$14017[4], 10)
          };
        }
        return null;
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][parseBackgroundImages] = function() {
        return this[backgroundImages] || (this[backgroundImages] = local$$13491(this[css](backgroundImage)));
      };
      /**
       * @param {?} local$$14079
       * @param {number} local$$14080
       * @return {?}
       */
      local$$13221[prototype][cssList] = function(local$$14079, local$$14080) {
        var local$$14095 = (this[css](local$$14079) || )[split](,);
        local$$14095 = local$$14095[local$$14080 || 0] || local$$14095[0] || auto;
        local$$14095 = local$$14095[trim]()[split]( );
        if (local$$14095[length] === 1) {
          /** @type {!Array} */
          local$$14095 = [local$$14095[0], local$$13436(local$$14095[0]) ? auto : local$$14095[0]];
        }
        return local$$14095;
      };
      /**
       * @param {?} local$$14152
       * @param {?} local$$14153
       * @param {?} local$$14154
       * @return {?}
       */
      local$$13221[prototype][parseBackgroundSize] = function(local$$14152, local$$14153, local$$14154) {
        var local$$14162 = this[cssList](backgroundSize, local$$14154);
        var local$$14164;
        var local$$14166;
        if (local$$13436(local$$14162[0])) {
          /** @type {number} */
          local$$14164 = local$$14152[width] * parseFloat(local$$14162[0]) / 100;
        } else {
          if (/contain|cover/[test](local$$14162[0])) {
            /** @type {number} */
            var local$$14198 = local$$14152[width] / local$$14152[height];
            /** @type {number} */
            var local$$14207 = local$$14153[width] / local$$14153[height];
            return local$$14198 < local$$14207 ^ local$$14162[0] === contain ? {
              width : local$$14152[height] * local$$14207,
              height : local$$14152[height]
            } : {
              width : local$$14152[width],
              height : local$$14152[width] / local$$14207
            };
          } else {
            /** @type {number} */
            local$$14164 = parseInt(local$$14162[0], 10);
          }
        }
        if (local$$14162[0] === auto && local$$14162[1] === auto) {
          local$$14166 = local$$14153[height];
        } else {
          if (local$$14162[1] === auto) {
            /** @type {number} */
            local$$14166 = local$$14164 / local$$14153[width] * local$$14153[height];
          } else {
            if (local$$13436(local$$14162[1])) {
              /** @type {number} */
              local$$14166 = local$$14152[height] * parseFloat(local$$14162[1]) / 100;
            } else {
              /** @type {number} */
              local$$14166 = parseInt(local$$14162[1], 10);
            }
          }
        }
        if (local$$14162[0] === auto) {
          /** @type {number} */
          local$$14164 = local$$14166 / local$$14153[height] * local$$14153[width];
        }
        return {
          width : local$$14164,
          height : local$$14166
        };
      };
      /**
       * @param {?} local$$14337
       * @param {boolean} local$$14338
       * @param {?} local$$14339
       * @param {number} local$$14340
       * @return {?}
       */
      local$$13221[prototype][parseBackgroundPosition] = function(local$$14337, local$$14338, local$$14339, local$$14340) {
        var local$$14348 = this[cssList](backgroundPosition, local$$14339);
        var local$$14350;
        var local$$14352;
        if (local$$13436(local$$14348[0])) {
          /** @type {number} */
          local$$14350 = (local$$14337[width] - (local$$14340 || local$$14338)[width]) * (parseFloat(local$$14348[0]) / 100);
        } else {
          /** @type {number} */
          local$$14350 = parseInt(local$$14348[0], 10);
        }
        if (local$$14348[1] === auto) {
          /** @type {number} */
          local$$14352 = local$$14350 / local$$14338[width] * local$$14338[height];
        } else {
          if (local$$13436(local$$14348[1])) {
            /** @type {number} */
            local$$14352 = (local$$14337[height] - (local$$14340 || local$$14338)[height]) * parseFloat(local$$14348[1]) / 100;
          } else {
            /** @type {number} */
            local$$14352 = parseInt(local$$14348[1], 10);
          }
        }
        if (local$$14348[0] === auto) {
          /** @type {number} */
          local$$14350 = local$$14352 / local$$14338[height] * local$$14338[width];
        }
        return {
          left : local$$14350,
          top : local$$14352
        };
      };
      /**
       * @param {?} local$$14460
       * @return {?}
       */
      local$$13221[prototype][parseBackgroundRepeat] = function(local$$14460) {
        return this[cssList](backgroundRepeat, local$$14460)[0];
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][parseTextShadows] = function() {
        var local$$14488 = this[css](textShadow);
        /** @type {!Array} */
        var local$$14491 = [];
        if (local$$14488 && local$$14488 !== none) {
          var local$$14502 = local$$14488[match](this.TEXT_SHADOW_PROPERTY);
          /** @type {number} */
          var local$$14505 = 0;
          for (; local$$14502 && local$$14505 < local$$14502[length]; local$$14505++) {
            var local$$14520 = local$$14502[local$$14505][match](this.TEXT_SHADOW_VALUES);
            local$$14491[push]({
              color : new local$$13476(local$$14520[0]),
              offsetX : local$$14520[1] ? parseFloat(local$$14520[1][replace](px, )) : 0,
              offsetY : local$$14520[2] ? parseFloat(local$$14520[2][replace](px, )) : 0,
              blur : local$$14520[3] ? local$$14520[3][replace](px, ) : 0
            });
          }
        }
        return local$$14491;
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][parseTransform] = function() {
        if (!this[transformData]) {
          if (this[hasTransform]()) {
            var local$$14604 = this[parseBounds]();
            var local$$14626 = this[prefixedCss](transformOrigin)[split]( )[map](local$$13452)[map](local$$13466);
            local$$14626[0] += local$$14604[left];
            local$$14626[1] += local$$14604[top];
            this[transformData] = {
              origin : local$$14626,
              matrix : this[parseTransformMatrix]()
            };
          } else {
            this[transformData] = {
              origin : [0, 0],
              matrix : [1, 0, 0, 1, 0, 0]
            };
          }
        }
        return this[transformData];
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][parseTransformMatrix] = function() {
        if (!this[transformMatrix]) {
          var local$$14699 = this[prefixedCss](transform);
          var local$$14709 = local$$14699 ? local$$13353(local$$14699[match](this.MATRIX_PROPERTY)) : null;
          this[transformMatrix] = local$$14709 ? local$$14709 : [1, 0, 0, 1, 0, 0];
        }
        return this[transformMatrix];
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][parseBounds] = function() {
        return this[bounds] || (this[bounds] = this[hasTransform]() ? local$$13496(this[node]) : local$$13486(this[node]));
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][hasTransform] = function() {
        return this[parseTransformMatrix]()[join](,) !== 1,0,0,1,0,0 || this[parent] && this[parent][hasTransform]();
      };
      /**
       * @return {?}
       */
      local$$13221[prototype][getValue] = function() {
        var local$$14821 = this[node][value] || ;
        if (this[node][tagName] === SELECT) {
          local$$14821 = local$$13327(this[node]);
        } else {
          if (this[node][type] === password) {
            local$$14821 = Array(local$$14821[length] + 1)[join](•);
          }
        }
        return local$$14821[length] === 0 ? this[node][placeholder] ||  : local$$14821;
      };
      /** @type {!RegExp} */
      local$$13221[prototype][MATRIX_PROPERTY] = /(matrix|matrix3d)\((.+)\)/;
      /** @type {!RegExp} */
      local$$13221[prototype][TEXT_SHADOW_PROPERTY] = /((rgba|rgb)\([^\)]+\)(\s-?\d+px){0,})/g;
      /** @type {!RegExp} */
      local$$13221[prototype][TEXT_SHADOW_VALUES] = /(-?\d+px)|(#.+)|(rgb\(.+\))|(rgba\(.+\))/g;
      /** @type {!RegExp} */
      local$$13221[prototype][CLIP] = /^rect\((\d+)px,? (\d+)px,? (\d+)px,? (\d+)px\)$/;
      /** @type {function(?, ?): undefined} */
      local$$13218[exports] = local$$13221;
    }, {
      "./color" : 3,
      "./utils" : 26
    }],
    15 : [function(local$$14939, local$$14940, local$$14941) {
      /**
       * @param {?} local$$14944
       * @param {?} local$$14945
       * @param {?} local$$14946
       * @param {?} local$$14947
       * @param {?} local$$14948
       * @return {undefined}
       */
      function local$$14943(local$$14944, local$$14945, local$$14946, local$$14947, local$$14948) {
        local$$14950(Starting NodeParser);
        this[renderer] = local$$14945;
        this[options] = local$$14948;
        /** @type {null} */
        this[range] = null;
        this[support] = local$$14946;
        /** @type {!Array} */
        this[renderQueue] = [];
        this[stack] = new local$$14985(true, 1, local$$14944[ownerDocument], null);
        var local$$14998 = new local$$14995(local$$14944, null);
        if (local$$14948[background]) {
          local$$14945[rectangle](0, 0, local$$14945[width], local$$14945[height], new local$$15014(local$$14948[background]));
        }
        if (local$$14944 === local$$14944[ownerDocument][documentElement]) {
          var local$$15056 = new local$$14995(local$$14998[color](backgroundColor)[isTransparent]() ? local$$14944[ownerDocument][body] : local$$14944[ownerDocument][documentElement], null);
          local$$14945[rectangle](0, 0, local$$14945[width], local$$14945[height], local$$15056[color](backgroundColor));
        }
        local$$14998[visibile] = local$$14998[isElementVisible]();
        this[createPseudoHideStyles](local$$14944[ownerDocument]);
        this[disableAnimations](local$$14944[ownerDocument]);
        this[nodes] = local$$15108([local$$14998][concat](this[getChildren](local$$14998))[filter](function(local$$15121) {
          return local$$15121[visible] = local$$15121[isElementVisible]();
        })[map](this[getPseudoElements], this));
        this[fontMetrics] = new local$$15148;
        local$$14950(Fetched nodes, total:, this[nodes][length]);
        local$$14950(Calculate overflow clips);
        this[calculateOverflowClips]();
        local$$14950(Start fetching images);
        this[images] = local$$14947[fetch](this[nodes][filter](local$$15187));
        this[ready] = this[images][ready][then](local$$15204(function() {
          local$$14950(Images loaded, starting parsing);
          local$$14950(Creating stacking contexts);
          this[createStackingContexts]();
          local$$14950(Sorting stacking contexts);
          this[sortStackingContexts](this[stack]);
          this[parse](this[stack]);
          local$$14950(Render queue created with  + this[renderQueue][length] +  items);
          return new Promise(local$$15204(function(local$$15253) {
            if (!local$$14948[async]) {
              this[renderQueue][forEach](this[paint], this);
              local$$15253();
            } else {
              if (typeof local$$14948[async] === function) {
                local$$14948[async][call](this, this[renderQueue], local$$15253);
              } else {
                if (this[renderQueue][length] > 0) {
                  /** @type {number} */
                  this[renderIndex] = 0;
                  this[asyncRenderer](this[renderQueue], local$$15253);
                } else {
                  local$$15253();
                }
              }
            }
          }, this));
        }, this));
      }
      /**
       * @param {?} local$$15337
       * @return {?}
       */
      function local$$15336(local$$15337) {
        return local$$15337[parent] && local$$15337[parent][clip][length];
      }
      /**
       * @param {?} local$$15356
       * @return {?}
       */
      function local$$15355(local$$15356) {
        return local$$15356[replace](/(\-[a-z])/g, function(local$$15363) {
          return local$$15363[toUpperCase]()[replace](-, );
        });
      }
      /**
       * @return {undefined}
       */
      function local$$15384() {
      }
      /**
       * @param {!Array} local$$15389
       * @param {?} local$$15390
       * @param {?} local$$15391
       * @param {!Array} local$$15392
       * @return {?}
       */
      function local$$15388(local$$15389, local$$15390, local$$15391, local$$15392) {
        return local$$15389[map](function(local$$15397, local$$15398) {
          if (local$$15397[width] > 0) {
            var local$$15408 = local$$15390[left];
            var local$$15413 = local$$15390[top];
            var local$$15418 = local$$15390[width];
            /** @type {number} */
            var local$$15429 = local$$15390[height] - local$$15389[2][width];
            switch(local$$15398) {
              case 0:
                local$$15429 = local$$15389[0][width];
                local$$15397[args] = local$$15442({
                  c1 : [local$$15408, local$$15413],
                  c2 : [local$$15408 + local$$15418, local$$15413],
                  c3 : [local$$15408 + local$$15418 - local$$15389[1][width], local$$15413 + local$$15429],
                  c4 : [local$$15408 + local$$15389[3][width], local$$15413 + local$$15429]
                }, local$$15392[0], local$$15392[1], local$$15391[topLeftOuter], local$$15391[topLeftInner], local$$15391[topRightOuter], local$$15391[topRightInner]);
                break;
              case 1:
                /** @type {number} */
                local$$15408 = local$$15390[left] + local$$15390[width] - local$$15389[1][width];
                local$$15418 = local$$15389[1][width];
                local$$15397[args] = local$$15442({
                  c1 : [local$$15408 + local$$15418, local$$15413],
                  c2 : [local$$15408 + local$$15418, local$$15413 + local$$15429 + local$$15389[2][width]],
                  c3 : [local$$15408, local$$15413 + local$$15429],
                  c4 : [local$$15408, local$$15413 + local$$15389[0][width]]
                }, local$$15392[1], local$$15392[2], local$$15391[topRightOuter], local$$15391[topRightInner], local$$15391[bottomRightOuter], local$$15391[bottomRightInner]);
                break;
              case 2:
                /** @type {number} */
                local$$15413 = local$$15413 + local$$15390[height] - local$$15389[2][width];
                local$$15429 = local$$15389[2][width];
                local$$15397[args] = local$$15442({
                  c1 : [local$$15408 + local$$15418, local$$15413 + local$$15429],
                  c2 : [local$$15408, local$$15413 + local$$15429],
                  c3 : [local$$15408 + local$$15389[3][width], local$$15413],
                  c4 : [local$$15408 + local$$15418 - local$$15389[3][width], local$$15413]
                }, local$$15392[2], local$$15392[3], local$$15391[bottomRightOuter], local$$15391[bottomRightInner], local$$15391[bottomLeftOuter], local$$15391[bottomLeftInner]);
                break;
              case 3:
                local$$15418 = local$$15389[3][width];
                local$$15397[args] = local$$15442({
                  c1 : [local$$15408, local$$15413 + local$$15429 + local$$15389[2][width]],
                  c2 : [local$$15408, local$$15413],
                  c3 : [local$$15408 + local$$15418, local$$15413 + local$$15389[0][width]],
                  c4 : [local$$15408 + local$$15418, local$$15413 + local$$15429]
                }, local$$15392[3], local$$15392[0], local$$15391[bottomLeftOuter], local$$15391[bottomLeftInner], local$$15391[topLeftOuter], local$$15391[topLeftInner]);
                break;
            }
          }
          return local$$15397;
        });
      }
      /**
       * @param {number} local$$15687
       * @param {number} local$$15688
       * @param {number} local$$15689
       * @param {number} local$$15690
       * @return {?}
       */
      function local$$15686(local$$15687, local$$15688, local$$15689, local$$15690) {
        /** @type {number} */
        var local$$15703 = 4 * ((Math[sqrt](2) - 1) / 3);
        /** @type {number} */
        var local$$15706 = local$$15689 * local$$15703;
        /** @type {number} */
        var local$$15709 = local$$15690 * local$$15703;
        var local$$15712 = local$$15687 + local$$15689;
        var local$$15715 = local$$15688 + local$$15690;
        return {
          topLeft : local$$15717({
            x : local$$15687,
            y : local$$15715
          }, {
            x : local$$15687,
            y : local$$15715 - local$$15709
          }, {
            x : local$$15712 - local$$15706,
            y : local$$15688
          }, {
            x : local$$15712,
            y : local$$15688
          }),
          topRight : local$$15717({
            x : local$$15687,
            y : local$$15688
          }, {
            x : local$$15687 + local$$15706,
            y : local$$15688
          }, {
            x : local$$15712,
            y : local$$15715 - local$$15709
          }, {
            x : local$$15712,
            y : local$$15715
          }),
          bottomRight : local$$15717({
            x : local$$15712,
            y : local$$15688
          }, {
            x : local$$15712,
            y : local$$15688 + local$$15709
          }, {
            x : local$$15687 + local$$15706,
            y : local$$15715
          }, {
            x : local$$15687,
            y : local$$15715
          }),
          bottomLeft : local$$15717({
            x : local$$15712,
            y : local$$15715
          }, {
            x : local$$15712 - local$$15706,
            y : local$$15715
          }, {
            x : local$$15687,
            y : local$$15688 + local$$15709
          }, {
            x : local$$15687,
            y : local$$15688
          })
        };
      }
      /**
       * @param {?} local$$15751
       * @param {!Array} local$$15752
       * @param {!Array} local$$15753
       * @return {?}
       */
      function local$$15750(local$$15751, local$$15752, local$$15753) {
        var local$$15758 = local$$15751[left];
        var local$$15763 = local$$15751[top];
        var local$$15768 = local$$15751[width];
        var local$$15773 = local$$15751[height];
        var local$$15789 = local$$15752[0][0] < local$$15768 / 2 ? local$$15752[0][0] : local$$15768 / 2;
        var local$$15805 = local$$15752[0][1] < local$$15773 / 2 ? local$$15752[0][1] : local$$15773 / 2;
        var local$$15821 = local$$15752[1][0] < local$$15768 / 2 ? local$$15752[1][0] : local$$15768 / 2;
        var local$$15837 = local$$15752[1][1] < local$$15773 / 2 ? local$$15752[1][1] : local$$15773 / 2;
        var local$$15853 = local$$15752[2][0] < local$$15768 / 2 ? local$$15752[2][0] : local$$15768 / 2;
        var local$$15869 = local$$15752[2][1] < local$$15773 / 2 ? local$$15752[2][1] : local$$15773 / 2;
        var local$$15885 = local$$15752[3][0] < local$$15768 / 2 ? local$$15752[3][0] : local$$15768 / 2;
        var local$$15901 = local$$15752[3][1] < local$$15773 / 2 ? local$$15752[3][1] : local$$15773 / 2;
        /** @type {number} */
        var local$$15904 = local$$15768 - local$$15821;
        /** @type {number} */
        var local$$15907 = local$$15773 - local$$15869;
        /** @type {number} */
        var local$$15910 = local$$15768 - local$$15853;
        /** @type {number} */
        var local$$15913 = local$$15773 - local$$15901;
        return {
          topLeftOuter : local$$15686(local$$15758, local$$15763, local$$15789, local$$15805)[topLeft][subdivide](.5),
          topLeftInner : local$$15686(local$$15758 + local$$15753[3][width], local$$15763 + local$$15753[0][width], Math[max](0, local$$15789 - local$$15753[3][width]), Math[max](0, local$$15805 - local$$15753[0][width]))[topLeft][subdivide](.5),
          topRightOuter : local$$15686(local$$15758 + local$$15904, local$$15763, local$$15821, local$$15837)[topRight][subdivide](.5),
          topRightInner : local$$15686(local$$15758 + Math[min](local$$15904, local$$15768 + local$$15753[3][width]), local$$15763 + local$$15753[0][width], local$$15904 > local$$15768 + local$$15753[3][width] ? 0 : local$$15821 - local$$15753[3][width], local$$15837 - local$$15753[0][width])[topRight][subdivide](.5),
          bottomRightOuter : local$$15686(local$$15758 + local$$15910, local$$15763 + local$$15907, local$$15853, local$$15869)[bottomRight][subdivide](.5),
          bottomRightInner : local$$15686(local$$15758 + Math[min](local$$15910, local$$15768 - local$$15753[3][width]), local$$15763 + Math[min](local$$15907, local$$15773 + local$$15753[0][width]), Math[max](0, local$$15853 - local$$15753[1][width]), local$$15869 - local$$15753[2][width])[bottomRight][subdivide](.5),
          bottomLeftOuter : local$$15686(local$$15758, local$$15763 + local$$15913, local$$15885, local$$15901)[bottomLeft][subdivide](.5),
          bottomLeftInner : local$$15686(local$$15758 + local$$15753[3][width], local$$15763 + local$$15913, Math[max](0, local$$15885 - local$$15753[3][width]), local$$15901 - local$$15753[2][width])[bottomLeft][subdivide](.5)
        };
      }
      /**
       * @param {number} local$$16130
       * @param {undefined} local$$16131
       * @param {undefined} local$$16132
       * @param {number} local$$16133
       * @return {?}
       */
      function local$$15717(local$$16130, local$$16131, local$$16132, local$$16133) {
        /**
         * @param {number} local$$16135
         * @param {number} local$$16136
         * @param {number} local$$16137
         * @return {?}
         */
        var local$$16167 = function(local$$16135, local$$16136, local$$16137) {
          return {
            x : local$$16135[x] + (local$$16136[x] - local$$16135[x]) * local$$16137,
            y : local$$16135[y] + (local$$16136[y] - local$$16135[y]) * local$$16137
          };
        };
        return {
          start : local$$16130,
          startControl : local$$16131,
          endControl : local$$16132,
          end : local$$16133,
          subdivide : function(local$$16171) {
            var local$$16174 = local$$16167(local$$16130, local$$16131, local$$16171);
            var local$$16177 = local$$16167(local$$16131, local$$16132, local$$16171);
            var local$$16180 = local$$16167(local$$16132, local$$16133, local$$16171);
            var local$$16183 = local$$16167(local$$16174, local$$16177, local$$16171);
            var local$$16186 = local$$16167(local$$16177, local$$16180, local$$16171);
            var local$$16189 = local$$16167(local$$16183, local$$16186, local$$16171);
            return [local$$15717(local$$16130, local$$16174, local$$16183, local$$16189), local$$15717(local$$16189, local$$16186, local$$16180, local$$16133)];
          },
          curveTo : function(local$$16197) {
            local$$16197[push]([bezierCurve, local$$16131[x], local$$16131[y], local$$16132[x], local$$16132[y], local$$16133[x], local$$16133[y]]);
          },
          curveToReversed : function(local$$16227) {
            local$$16227[push]([bezierCurve, local$$16132[x], local$$16132[y], local$$16131[x], local$$16131[y], local$$16130[x], local$$16130[y]]);
          }
        };
      }
      /**
       * @param {?} local$$16261
       * @param {!Object} local$$16262
       * @param {!Object} local$$16263
       * @param {!Object} local$$16264
       * @param {!Object} local$$16265
       * @param {!Object} local$$16266
       * @param {!Object} local$$16267
       * @return {?}
       */
      function local$$15442(local$$16261, local$$16262, local$$16263, local$$16264, local$$16265, local$$16266, local$$16267) {
        /** @type {!Array} */
        var local$$16270 = [];
        if (local$$16262[0] > 0 || local$$16262[1] > 0) {
          local$$16270[push]([line, local$$16264[1][start][x], local$$16264[1][start][y]]);
          local$$16264[1][curveTo](local$$16270);
        } else {
          local$$16270[push]([line, local$$16261[c1][0], local$$16261[c1][1]]);
        }
        if (local$$16263[0] > 0 || local$$16263[1] > 0) {
          local$$16270[push]([line, local$$16266[0][start][x], local$$16266[0][start][y]]);
          local$$16266[0][curveTo](local$$16270);
          local$$16270[push]([line, local$$16267[0][end][x], local$$16267[0][end][y]]);
          local$$16267[0][curveToReversed](local$$16270);
        } else {
          local$$16270[push]([line, local$$16261[c2][0], local$$16261[c2][1]]);
          local$$16270[push]([line, local$$16261[c3][0], local$$16261[c3][1]]);
        }
        if (local$$16262[0] > 0 || local$$16262[1] > 0) {
          local$$16270[push]([line, local$$16265[1][end][x], local$$16265[1][end][y]]);
          local$$16265[1][curveToReversed](local$$16270);
        } else {
          local$$16270[push]([line, local$$16261[c4][0], local$$16261[c4][1]]);
        }
        return local$$16270;
      }
      /**
       * @param {!Array} local$$16511
       * @param {!Object} local$$16512
       * @param {!Object} local$$16513
       * @param {!Object} local$$16514
       * @param {!Object} local$$16515
       * @param {number} local$$16516
       * @param {number} local$$16517
       * @return {undefined}
       */
      function local$$16510(local$$16511, local$$16512, local$$16513, local$$16514, local$$16515, local$$16516, local$$16517) {
        if (local$$16512[0] > 0 || local$$16512[1] > 0) {
          local$$16511[push]([line, local$$16514[0][start][x], local$$16514[0][start][y]]);
          local$$16514[0][curveTo](local$$16511);
          local$$16514[1][curveTo](local$$16511);
        } else {
          local$$16511[push]([line, local$$16516, local$$16517]);
        }
        if (local$$16513[0] > 0 || local$$16513[1] > 0) {
          local$$16511[push]([line, local$$16515[0][start][x], local$$16515[0][start][y]]);
        }
      }
      /**
       * @param {?} local$$16616
       * @return {?}
       */
      function local$$16615(local$$16616) {
        return local$$16616[cssInt](zIndex) < 0;
      }
      /**
       * @param {?} local$$16630
       * @return {?}
       */
      function local$$16629(local$$16630) {
        return local$$16630[cssInt](zIndex) > 0;
      }
      /**
       * @param {?} local$$16644
       * @return {?}
       */
      function local$$16643(local$$16644) {
        return local$$16644[cssInt](zIndex) === 0;
      }
      /**
       * @param {?} local$$16658
       * @return {?}
       */
      function local$$16657(local$$16658) {
        return [inline, inline-block, inline-table][indexOf](local$$16658[css](display)) !== -1;
      }
      /**
       * @param {?} local$$16683
       * @return {?}
       */
      function local$$16682(local$$16683) {
        return local$$16683 instanceof local$$14985;
      }
      /**
       * @param {?} local$$16690
       * @return {?}
       */
      function local$$16689(local$$16690) {
        return local$$16690[node][data][trim]()[length] > 0;
      }
      /**
       * @param {?} local$$16711
       * @return {?}
       */
      function local$$16710(local$$16711) {
        return /^(normal|none|0px)$/[test](local$$16711[parent][css](letterSpacing));
      }
      /**
       * @param {?} local$$16732
       * @return {?}
       */
      function local$$16731(local$$16732) {
        return [TopLeft, TopRight, BottomRight, BottomLeft][map](function(local$$16746) {
          var local$$16758 = local$$16732[css](border + local$$16746 + Radius);
          var local$$16766 = local$$16758[split]( );
          if (local$$16766[length] <= 1) {
            local$$16766[1] = local$$16766[0];
          }
          return local$$16766[map](local$$16785);
        });
      }
      /**
       * @param {?} local$$16795
       * @return {?}
       */
      function local$$16794(local$$16795) {
        return local$$16795[nodeType] === Node[TEXT_NODE] || local$$16795[nodeType] === Node[ELEMENT_NODE];
      }
      /**
       * @param {?} local$$16816
       * @return {?}
       */
      function local$$16815(local$$16816) {
        var local$$16824 = local$$16816[css](position);
        var local$$16848 = [absolute, relative, fixed][indexOf](local$$16824) !== -1 ? local$$16816[css](zIndex) : auto;
        return local$$16848 !== auto;
      }
      /**
       * @param {?} local$$16857
       * @return {?}
       */
      function local$$16856(local$$16857) {
        return local$$16857[css](position) !== static;
      }
      /**
       * @param {?} local$$16872
       * @return {?}
       */
      function local$$16871(local$$16872) {
        return local$$16872[css](float) !== none;
      }
      /**
       * @param {?} local$$16887
       * @return {?}
       */
      function local$$16886(local$$16887) {
        return [inline-block, inline-table][indexOf](local$$16887[css](display)) !== -1;
      }
      /**
       * @param {!Function} local$$16910
       * @return {?}
       */
      function local$$16909(local$$16910) {
        var local$$16912 = this;
        return function() {
          return !local$$16910[apply](local$$16912, arguments);
        };
      }
      /**
       * @param {?} local$$16926
       * @return {?}
       */
      function local$$15187(local$$16926) {
        return local$$16926[node][nodeType] === Node[ELEMENT_NODE];
      }
      /**
       * @param {?} local$$16942
       * @return {?}
       */
      function local$$16941(local$$16942) {
        return local$$16942[isPseudoElement] === true;
      }
      /**
       * @param {?} local$$16953
       * @return {?}
       */
      function local$$16952(local$$16953) {
        return local$$16953[node][nodeType] === Node[TEXT_NODE];
      }
      /**
       * @param {?} local$$16969
       * @return {?}
       */
      function local$$16968(local$$16969) {
        return function(local$$16971, local$$16972) {
          return local$$16971[cssInt](zIndex) + local$$16969[indexOf](local$$16971) / local$$16969[length] - (local$$16972[cssInt](zIndex) + local$$16969[indexOf](local$$16972) / local$$16969[length]);
        };
      }
      /**
       * @param {?} local$$17012
       * @return {?}
       */
      function local$$17011(local$$17012) {
        return local$$17012[getOpacity]() < 1;
      }
      /**
       * @param {?} local$$17023
       * @return {?}
       */
      function local$$16785(local$$17023) {
        return parseInt(local$$17023, 10);
      }
      /**
       * @param {?} local$$17031
       * @return {?}
       */
      function local$$17030(local$$17031) {
        return local$$17031[width];
      }
      /**
       * @param {?} local$$17040
       * @return {?}
       */
      function local$$17039(local$$17040) {
        return local$$17040[node][nodeType] !== Node[ELEMENT_NODE] || [SCRIPT, HEAD, TITLE, OBJECT, BR, OPTION][indexOf](local$$17040[node][nodeName]) === -1;
      }
      /**
       * @param {!Array} local$$17081
       * @return {?}
       */
      function local$$15108(local$$17081) {
        return [][concat][apply]([], local$$17081);
      }
      /**
       * @param {?} local$$17096
       * @return {?}
       */
      function local$$17095(local$$17096) {
        var local$$17104 = local$$17096[substr](0, 1);
        return local$$17104 === local$$17096[substr](local$$17096[length] - 1) && local$$17104[match](/'|"/) ? local$$17096[substr](1, local$$17096[length] - 2) : local$$17096;
      }
      /**
       * @param {!NodeList} local$$17138
       * @return {?}
       */
      function local$$17137(local$$17138) {
        /** @type {!Array} */
        var local$$17141 = [];
        /** @type {number} */
        var local$$17144 = 0;
        /** @type {boolean} */
        var local$$17147 = false;
        var local$$17149;
        for (; local$$17138[length];) {
          if (local$$17156(local$$17138[local$$17144]) === local$$17147) {
            local$$17149 = local$$17138[splice](0, local$$17144);
            if (local$$17149[length]) {
              local$$17141[push](local$$17173[ucs2][encode](local$$17149));
            }
            /** @type {boolean} */
            local$$17147 = !local$$17147;
            /** @type {number} */
            local$$17144 = 0;
          } else {
            local$$17144++;
          }
          if (local$$17144 >= local$$17138[length]) {
            local$$17149 = local$$17138[splice](0, local$$17144);
            if (local$$17149[length]) {
              local$$17141[push](local$$17173[ucs2][encode](local$$17149));
            }
          }
        }
        return local$$17141;
      }
      /**
       * @param {?} local$$17234
       * @return {?}
       */
      function local$$17156(local$$17234) {
        return [32, 13, 10, 9, 45][indexOf](local$$17234) !== -1;
      }
      /**
       * @param {?} local$$17252
       * @return {?}
       */
      function local$$17251(local$$17252) {
        return /[^\u0000-\u00ff]/[test](local$$17252);
      }
      var local$$14950 = local$$14939(./log);
      var local$$17173 = local$$14939(punycode);
      var local$$14995 = local$$14939(./nodecontainer);
      var local$$17278 = local$$14939(./textcontainer);
      var local$$17283 = local$$14939(./pseudoelementcontainer);
      var local$$15148 = local$$14939(./fontmetrics);
      var local$$15014 = local$$14939(./color);
      var local$$14985 = local$$14939(./stackingcontext);
      var local$$17300 = local$$14939(./utils);
      var local$$15204 = local$$17300[bind];
      var local$$17309 = local$$17300[getBounds];
      var local$$17314 = local$$17300[parseBackgrounds];
      var local$$17319 = local$$17300[offsetBounds];
      /**
       * @return {undefined}
       */
      local$$14943[prototype][calculateOverflowClips] = function() {
        this[nodes][forEach](function(local$$17334) {
          if (local$$15187(local$$17334)) {
            if (local$$16941(local$$17334)) {
              local$$17334[appendToDOM]();
            }
            local$$17334[borders] = this[parseBorders](local$$17334);
            /** @type {!Array} */
            var local$$17373 = local$$17334[css](overflow) === hidden ? [local$$17334[borders][clip]] : [];
            var local$$17379 = local$$17334[parseClip]();
            if (local$$17379 && [absolute, fixed][indexOf](local$$17334[css](position)) !== -1) {
              local$$17373[push]([[rect, local$$17334[bounds][left] + local$$17379[left], local$$17334[bounds][top] + local$$17379[top], local$$17379[right] - local$$17379[left], local$$17379[bottom] - local$$17379[top]]]);
            }
            local$$17334[clip] = local$$15336(local$$17334) ? local$$17334[parent][clip][concat](local$$17373) : local$$17373;
            local$$17334[backgroundClip] = local$$17334[css](overflow) !== hidden ? local$$17334[clip][concat]([local$$17334[borders][clip]]) : local$$17334[clip];
            if (local$$16941(local$$17334)) {
              local$$17334[cleanDOM]();
            }
          } else {
            if (local$$16952(local$$17334)) {
              local$$17334[clip] = local$$15336(local$$17334) ? local$$17334[parent][clip] : [];
            }
          }
          if (!local$$16941(local$$17334)) {
            /** @type {null} */
            local$$17334[bounds] = null;
          }
        }, this);
      };
      /**
       * @param {!Object} local$$17547
       * @param {?} local$$17548
       * @param {number} local$$17549
       * @return {undefined}
       */
      local$$14943[prototype][asyncRenderer] = function(local$$17547, local$$17548, local$$17549) {
        local$$17549 = local$$17549 || Date[now]();
        this[paint](local$$17547[this[renderIndex]++]);
        if (local$$17547[length] === this[renderIndex]) {
          local$$17548();
        } else {
          if (local$$17549 + 20 > Date[now]()) {
            this[asyncRenderer](local$$17547, local$$17548, local$$17549);
          } else {
            setTimeout(local$$15204(function() {
              this[asyncRenderer](local$$17547, local$$17548);
            }, this), 0);
          }
        }
      };
      /**
       * @param {?} local$$17617
       * @return {undefined}
       */
      local$$14943[prototype][createPseudoHideStyles] = function(local$$17617) {
        this[createStyles](local$$17617, . + local$$17283[prototype][PSEUDO_HIDE_ELEMENT_CLASS_BEFORE] + :before { content: "" !important; display: none !important; } + . + local$$17283[prototype][PSEUDO_HIDE_ELEMENT_CLASS_AFTER] + :after { content: "" !important; display: none !important; });
      };
      /**
       * @param {?} local$$17659
       * @return {undefined}
       */
      local$$14943[prototype][disableAnimations] = function(local$$17659) {
        this[createStyles](local$$17659, * { -webkit-animation: none !important; -moz-animation: none !important; -o-animation: none !important; animation: none !important;  + -webkit-transition: none !important; -moz-transition: none !important; -o-transition: none !important; transition: none !important;});
      };
      /**
       * @param {?} local$$17681
       * @param {?} local$$17682
       * @return {undefined}
       */
      local$$14943[prototype][createStyles] = function(local$$17681, local$$17682) {
        var local$$17690 = local$$17681[createElement](style);
        local$$17690[innerHTML] = local$$17682;
        local$$17681[body][appendChild](local$$17690);
      };
      /**
       * @param {?} local$$17715
       * @return {?}
       */
      local$$14943[prototype][getPseudoElements] = function(local$$17715) {
        /** @type {!Array} */
        var local$$17719 = [[local$$17715]];
        if (local$$17715[node][nodeType] === Node[ELEMENT_NODE]) {
          var local$$17737 = this[getPseudoElement](local$$17715, :before);
          var local$$17745 = this[getPseudoElement](local$$17715, :after);
          if (local$$17737) {
            local$$17719[push](local$$17737);
          }
          if (local$$17745) {
            local$$17719[push](local$$17745);
          }
        }
        return local$$15108(local$$17719);
      };
      /**
       * @param {(ArrayBuffer|ArrayBufferView|Blob|string)} local$$17777
       * @param {undefined} local$$17778
       * @return {?}
       */
      local$$14943[prototype][getPseudoElement] = function(local$$17777, local$$17778) {
        var local$$17784 = local$$17777[computedStyle](local$$17778);
        if (!local$$17784 || !local$$17784[content] || local$$17784[content] === none || local$$17784[content] === -moz-alt-content || local$$17784[display] === none) {
          return null;
        }
        var local$$17822 = local$$17095(local$$17784[content]);
        /** @type {boolean} */
        var local$$17833 = local$$17822[substr](0, 3) === url;
        var local$$17844 = document[createElement](local$$17833 ? img : html2canvaspseudoelement);
        var local$$17847 = new local$$17283(local$$17844, local$$17777, local$$17778);
        /** @type {number} */
        var local$$17854 = local$$17784[length] - 1;
        for (; local$$17854 >= 0; local$$17854--) {
          var local$$17865 = local$$15355(local$$17784[item](local$$17854));
          local$$17844[style][local$$17865] = local$$17784[local$$17865];
        }
        local$$17844[className] = local$$17283[prototype][PSEUDO_HIDE_ELEMENT_CLASS_BEFORE] +   + local$$17283[prototype][PSEUDO_HIDE_ELEMENT_CLASS_AFTER];
        if (local$$17833) {
          local$$17844[src] = local$$17314(local$$17822)[0][args][0];
          return [local$$17847];
        } else {
          var local$$17918 = document[createTextNode](local$$17822);
          local$$17844[appendChild](local$$17918);
          return [local$$17847, new local$$17278(local$$17918, local$$17847)];
        }
      };
      /**
       * @param {?} local$$17940
       * @return {?}
       */
      local$$14943[prototype][getChildren] = function(local$$17940) {
        return local$$15108([][filter][call](local$$17940[node][childNodes], local$$16794)[map](function(local$$17959) {
          var local$$17976 = [local$$17959[nodeType] === Node[TEXT_NODE] ? new local$$17278(local$$17959, local$$17940) : new local$$14995(local$$17959, local$$17940)][filter](local$$17039);
          return local$$17959[nodeType] === Node[ELEMENT_NODE] && local$$17976[length] && local$$17959[tagName] !== TEXTAREA ? local$$17976[0][isElementVisible]() ? local$$17976[concat](this[getChildren](local$$17976[0])) : [] : local$$17976;
        }, this));
      };
      /**
       * @param {?} local$$18031
       * @param {?} local$$18032
       * @return {undefined}
       */
      local$$14943[prototype][newStackingContext] = function(local$$18031, local$$18032) {
        var local$$18045 = new local$$14985(local$$18032, local$$18031[getOpacity](), local$$18031[node], local$$18031[parent]);
        local$$18031[cloneTo](local$$18045);
        var local$$18063 = local$$18032 ? local$$18045[getParentStack](this) : local$$18045[parent][stack];
        local$$18063[contexts][push](local$$18045);
        local$$18031[stack] = local$$18045;
      };
      /**
       * @return {undefined}
       */
      local$$14943[prototype][createStackingContexts] = function() {
        this[nodes][forEach](function(local$$18095) {
          if (local$$15187(local$$18095) && (this[isRootElement](local$$18095) || local$$17011(local$$18095) || local$$16815(local$$18095) || this[isBodyWithTransparentRoot](local$$18095) || local$$18095[hasTransform]())) {
            this[newStackingContext](local$$18095, true);
          } else {
            if (local$$15187(local$$18095) && (local$$16856(local$$18095) && local$$16643(local$$18095) || local$$16886(local$$18095) || local$$16871(local$$18095))) {
              this[newStackingContext](local$$18095, false);
            } else {
              local$$18095[assignStack](local$$18095[parent][stack]);
            }
          }
        }, this);
      };
      /**
       * @param {?} local$$18169
       * @return {?}
       */
      local$$14943[prototype][isBodyWithTransparentRoot] = function(local$$18169) {
        return local$$18169[node][nodeName] === BODY && local$$18169[parent][color](backgroundColor)[isTransparent]();
      };
      /**
       * @param {?} local$$18205
       * @return {?}
       */
      local$$14943[prototype][isRootElement] = function(local$$18205) {
        return local$$18205[parent] === null;
      };
      /**
       * @param {?} local$$18223
       * @return {undefined}
       */
      local$$14943[prototype][sortStackingContexts] = function(local$$18223) {
        local$$18223[contexts][sort](local$$16968(local$$18223[contexts][slice](0)));
        local$$18223[contexts][forEach](this[sortStackingContexts], this);
      };
      /**
       * @param {?} local$$18263
       * @return {?}
       */
      local$$14943[prototype][parseTextBounds] = function(local$$18263) {
        return function(local$$18265, local$$18266, local$$18267) {
          if (local$$18263[parent][css](textDecoration)[substr](0, 4) !== none || local$$18265[trim]()[length] !== 0) {
            if (this[support][rangeBounds] && !local$$18263[parent][hasTransform]()) {
              var local$$18326 = local$$18267[slice](0, local$$18266)[join]()[length];
              return this[getRangeBounds](local$$18263[node], local$$18326, local$$18265[length]);
            } else {
              if (local$$18263[node] && typeof local$$18263[node][data] === string) {
                var local$$18364 = local$$18263[node][splitText](local$$18265[length]);
                var local$$18380 = this[getWrapperBounds](local$$18263[node], local$$18263[parent][hasTransform]());
                local$$18263[node] = local$$18364;
                return local$$18380;
              }
            }
          } else {
            if (!this[support][rangeBounds] || local$$18263[parent][hasTransform]()) {
              local$$18263[node] = local$$18263[node][splitText](local$$18265[length]);
            }
          }
          return {};
        };
      };
      /**
       * @param {?} local$$18443
       * @param {?} local$$18444
       * @return {?}
       */
      local$$14943[prototype][getWrapperBounds] = function(local$$18443, local$$18444) {
        var local$$18455 = local$$18443[ownerDocument][createElement](html2canvaswrapper);
        var local$$18460 = local$$18443[parentNode];
        var local$$18467 = local$$18443[cloneNode](true);
        local$$18455[appendChild](local$$18443[cloneNode](true));
        local$$18460[replaceChild](local$$18455, local$$18443);
        var local$$18487 = local$$18444 ? local$$17319(local$$18455) : local$$17309(local$$18455);
        local$$18460[replaceChild](local$$18467, local$$18455);
        return local$$18487;
      };
      /**
       * @param {?} local$$18505
       * @param {(Object|number)} local$$18506
       * @param {!Object} local$$18507
       * @return {?}
       */
      local$$14943[prototype][getRangeBounds] = function(local$$18505, local$$18506, local$$18507) {
        var local$$18524 = this[range] || (this[range] = local$$18505[ownerDocument][createRange]());
        local$$18524[setStart](local$$18505, local$$18506);
        local$$18524[setEnd](local$$18505, local$$18506 + local$$18507);
        return local$$18524[getBoundingClientRect]();
      };
      /**
       * @param {?} local$$18552
       * @return {undefined}
       */
      local$$14943[prototype][parse] = function(local$$18552) {
        var local$$18561 = local$$18552[contexts][filter](local$$16615);
        var local$$18570 = local$$18552[children][filter](local$$15187);
        var local$$18577 = local$$18570[filter](local$$16909(local$$16871));
        var local$$18589 = local$$18577[filter](local$$16909(local$$16856))[filter](local$$16909(local$$16657));
        var local$$18600 = local$$18570[filter](local$$16909(local$$16856))[filter](local$$16871);
        var local$$18611 = local$$18577[filter](local$$16909(local$$16856))[filter](local$$16657);
        var local$$18628 = local$$18552[contexts][concat](local$$18577[filter](local$$16856))[filter](local$$16643);
        var local$$18641 = local$$18552[children][filter](local$$16952)[filter](local$$16689);
        var local$$18650 = local$$18552[contexts][filter](local$$16629);
        local$$18561[concat](local$$18589)[concat](local$$18600)[concat](local$$18611)[concat](local$$18628)[concat](local$$18641)[concat](local$$18650)[forEach](function(local$$18679) {
          this[renderQueue][push](local$$18679);
          if (local$$16682(local$$18679)) {
            this[parse](local$$18679);
            this[renderQueue][push](new local$$15384);
          }
        }, this);
      };
      /**
       * @param {?} local$$18720
       * @return {undefined}
       */
      local$$14943[prototype][paint] = function(local$$18720) {
        try {
          if (local$$18720 instanceof local$$15384) {
            this[renderer][ctx][restore]();
          } else {
            if (local$$16952(local$$18720)) {
              if (local$$16941(local$$18720[parent])) {
                local$$18720[parent][appendToDOM]();
              }
              this[paintText](local$$18720);
              if (local$$16941(local$$18720[parent])) {
                local$$18720[parent][cleanDOM]();
              }
            } else {
              this[paintNode](local$$18720);
            }
          }
        } catch (local$$18781) {
          local$$14950(local$$18781);
          if (this[options][strict]) {
            throw local$$18781;
          }
        }
      };
      /**
       * @param {?} local$$18807
       * @return {undefined}
       */
      local$$14943[prototype][paintNode] = function(local$$18807) {
        if (local$$16682(local$$18807)) {
          this[renderer][setOpacity](local$$18807[opacity]);
          this[renderer][ctx][save]();
          if (local$$18807[hasTransform]()) {
            this[renderer][setTransform](local$$18807[parseTransform]());
          }
        }
        if (local$$18807[node][nodeName] === INPUT && local$$18807[node][type] === checkbox) {
          this[paintCheckbox](local$$18807);
        } else {
          if (local$$18807[node][nodeName] === INPUT && local$$18807[node][type] === radio) {
            this[paintRadio](local$$18807);
          } else {
            this[paintElement](local$$18807);
          }
        }
      };
      /**
       * @param {?} local$$18922
       * @return {undefined}
       */
      local$$14943[prototype][paintElement] = function(local$$18922) {
        var local$$18928 = local$$18922[parseBounds]();
        this[renderer][clip](local$$18922[backgroundClip], function() {
          this[renderer][renderBackground](local$$18922, local$$18928, local$$18922[borders][borders][map](local$$17030));
        }, this);
        this[renderer][clip](local$$18922[clip], function() {
          this[renderer][renderBorders](local$$18922[borders][borders]);
        }, this);
        this[renderer][clip](local$$18922[backgroundClip], function() {
          switch(local$$18922[node][nodeName]) {
            case svg:
            case IFRAME:
              var local$$19023 = this[images][get](local$$18922[node]);
              if (local$$19023) {
                this[renderer][renderImage](local$$18922, local$$18928, local$$18922[borders], local$$19023);
              } else {
                local$$14950(Error loading < + local$$18922[node][nodeName] + >, local$$18922[node]);
              }
              break;
            case IMG:
              var local$$19075 = this[images][get](local$$18922[node][src]);
              if (local$$19075) {
                this[renderer][renderImage](local$$18922, local$$18928, local$$18922[borders], local$$19075);
              } else {
                local$$14950(Error loading <img>, local$$18922[node][src]);
              }
              break;
            case CANVAS:
              this[renderer][renderImage](local$$18922, local$$18928, local$$18922[borders], {
                image : local$$18922[node]
              });
              break;
            case SELECT:
            case INPUT:
            case TEXTAREA:
              this[paintFormValue](local$$18922);
              break;
          }
        }, this);
      };
      /**
       * @param {?} local$$19160
       * @return {undefined}
       */
      local$$14943[prototype][paintCheckbox] = function(local$$19160) {
        var local$$19166 = local$$19160[parseBounds]();
        var local$$19178 = Math[min](local$$19166[width], local$$19166[height]);
        var local$$19191 = {
          width : local$$19178 - 1,
          height : local$$19178 - 1,
          top : local$$19166[top],
          left : local$$19166[left]
        };
        /** @type {!Array} */
        var local$$19196 = [3, 3];
        /** @type {!Array} */
        var local$$19199 = [local$$19196, local$$19196, local$$19196, local$$19196];
        var local$$19219 = [1, 1, 1, 1][map](function(local$$19209) {
          return {
            color : new local$$15014(#A5A5A5),
            width : local$$19209
          };
        });
        var local$$19222 = local$$15750(local$$19191, local$$19199, local$$19219);
        this[renderer][clip](local$$19160[backgroundClip], function() {
          this[renderer][rectangle](local$$19191[left] + 1, local$$19191[top] + 1, local$$19191[width] - 2, local$$19191[height] - 2, new local$$15014(#DEDEDE));
          this[renderer][renderBorders](local$$15388(local$$19219, local$$19191, local$$19222, local$$19199));
          if (local$$19160[node][checked]) {
            this[renderer][font](new local$$15014(#424242), normal, normal, bold, local$$19178 - 3 + px, arial);
            this[renderer][text](✔, local$$19191[left] + local$$19178 / 6, local$$19191[top] + local$$19178 - 1);
          }
        }, this);
      };
      /**
       * @param {?} local$$19342
       * @return {undefined}
       */
      local$$14943[prototype][paintRadio] = function(local$$19342) {
        var local$$19348 = local$$19342[parseBounds]();
        /** @type {number} */
        var local$$19362 = Math[min](local$$19348[width], local$$19348[height]) - 2;
        this[renderer][clip](local$$19342[backgroundClip], function() {
          this[renderer][circleStroke](local$$19348[left] + 1, local$$19348[top] + 1, local$$19362, new local$$15014(#DEDEDE), 1, new local$$15014(#A5A5A5));
          if (local$$19342[node][checked]) {
            this[renderer][circle](Math[ceil](local$$19348[left] + local$$19362 / 4) + 1, Math[ceil](local$$19348[top] + local$$19362 / 4) + 1, Math[floor](local$$19362 / 2), new local$$15014(#424242));
          }
        }, this);
      };
      /**
       * @param {?} local$$19462
       * @return {undefined}
       */
      local$$14943[prototype][paintFormValue] = function(local$$19462) {
        var local$$19468 = local$$19462[getValue]();
        if (local$$19468[length] > 0) {
          var local$$19481 = local$$19462[node][ownerDocument];
          var local$$19489 = local$$19481[createElement](html2canvaswrapper);
          /** @type {!Array} */
          var local$$19530 = [lineHeight, textAlign, fontFamily, fontWeight, fontSize, color, paddingLeft, paddingTop, paddingRight, paddingBottom, width, height, borderLeftStyle, borderTopStyle, borderLeftWidth, borderTopWidth, boxSizing, whiteSpace, wordWrap];
          local$$19530[forEach](function(local$$19535) {
            try {
              local$$19489[style][local$$19535] = local$$19462[css](local$$19535);
            } catch (local$$19548) {
              local$$14950(html2canvas: Parse: Exception caught in renderFormValue:  + local$$19548[message]);
            }
          });
          var local$$19569 = local$$19462[parseBounds]();
          local$$19489[style][position] = fixed;
          local$$19489[style][left] = local$$19569[left] + px;
          local$$19489[style][top] = local$$19569[top] + px;
          local$$19489[textContent] = local$$19468;
          local$$19481[body][appendChild](local$$19489);
          this[paintText](new local$$17278(local$$19489[firstChild], local$$19462));
          local$$19481[body][removeChild](local$$19489);
        }
      };
      /**
       * @param {?} local$$19651
       * @return {undefined}
       */
      local$$14943[prototype][paintText] = function(local$$19651) {
        local$$19651[applyTextTransform]();
        var local$$19671 = local$$17173[ucs2][decode](local$$19651[node][data]);
        var local$$19710 = (!this[options][letterRendering] || local$$16710(local$$19651)) && !local$$17251(local$$19651[node][data]) ? local$$17137(local$$19671) : local$$19671[map](function(local$$19695) {
          return local$$17173[ucs2][encode]([local$$19695]);
        });
        var local$$19719 = local$$19651[parent][fontWeight]();
        var local$$19730 = local$$19651[parent][css](fontSize);
        var local$$19741 = local$$19651[parent][css](fontFamily);
        var local$$19750 = local$$19651[parent][parseTextShadows]();
        this[renderer][font](local$$19651[parent][color](color), local$$19651[parent][css](fontStyle), local$$19651[parent][css](fontVariant), local$$19719, local$$19730, local$$19741);
        if (local$$19750[length]) {
          this[renderer][fontShadow](local$$19750[0][color], local$$19750[0][offsetX], local$$19750[0][offsetY], local$$19750[0][blur]);
        } else {
          this[renderer][clearShadow]();
        }
        this[renderer][clip](local$$19651[parent][clip], function() {
          local$$19710[map](this[parseTextBounds](local$$19651), this)[forEach](function(local$$19854, local$$19855) {
            if (local$$19854) {
              this[renderer][text](local$$19710[local$$19855], local$$19854[left], local$$19854[bottom]);
              this[renderTextDecoration](local$$19651[parent], local$$19854, this[fontMetrics][getMetrics](local$$19741, local$$19730));
            }
          }, this);
        }, this);
      };
      /**
       * @param {?} local$$19907
       * @param {?} local$$19908
       * @param {?} local$$19909
       * @return {undefined}
       */
      local$$14943[prototype][renderTextDecoration] = function(local$$19907, local$$19908, local$$19909) {
        switch(local$$19907[css](textDecoration)[split]( )[0]) {
          case underline:
            this[renderer][rectangle](local$$19908[left], Math[round](local$$19908[top] + local$$19909[baseline] + local$$19909[lineWidth]), local$$19908[width], 1, local$$19907[color](color));
            break;
          case overline:
            this[renderer][rectangle](local$$19908[left], Math[round](local$$19908[top]), local$$19908[width], 1, local$$19907[color](color));
            break;
          case line-through:
            this[renderer][rectangle](local$$19908[left], Math[ceil](local$$19908[top] + local$$19909[middle] + local$$19909[lineWidth]), local$$19908[width], 1, local$$19907[color](color));
            break;
        }
      };
      var local$$20063 = {
        inset : [[darken, .6], [darken, .1], [darken, .1], [darken, .6]]
      };
      /**
       * @param {?} local$$20071
       * @return {?}
       */
      local$$14943[prototype][parseBorders] = function(local$$20071) {
        var local$$20077 = local$$20071[parseBounds]();
        var local$$20080 = local$$16731(local$$20071);
        var local$$20172 = [Top, Right, Bottom, Left][map](function(local$$20094, local$$20095) {
          var local$$20107 = local$$20071[css](border + local$$20094 + Style);
          var local$$20119 = local$$20071[color](border + local$$20094 + Color);
          if (local$$20107 === inset && local$$20119[isBlack]()) {
            local$$20119 = new local$$15014([255, 255, 255, local$$20119[a]]);
          }
          var local$$20147 = local$$20063[local$$20107] ? local$$20063[local$$20107][local$$20095] : null;
          return {
            width : local$$20071[cssInt](border + local$$20094 + Width),
            color : local$$20147 ? local$$20119[local$$20147[0]](local$$20147[1]) : local$$20119,
            args : null
          };
        });
        var local$$20175 = local$$15750(local$$20077, local$$20080, local$$20172);
        return {
          clip : this[parseBackgroundClip](local$$20071, local$$20175, local$$20172, local$$20080, local$$20077),
          borders : local$$15388(local$$20172, local$$20077, local$$20175, local$$20080)
        };
      };
      /**
       * @param {?} local$$20194
       * @param {?} local$$20195
       * @param {!Array} local$$20196
       * @param {!Array} local$$20197
       * @param {?} local$$20198
       * @return {?}
       */
      local$$14943[prototype][parseBackgroundClip] = function(local$$20194, local$$20195, local$$20196, local$$20197, local$$20198) {
        var local$$20206 = local$$20194[css](backgroundClip);
        /** @type {!Array} */
        var local$$20209 = [];
        switch(local$$20206) {
          case content-box:
          case padding-box:
            local$$16510(local$$20209, local$$20197[0], local$$20197[1], local$$20195[topLeftInner], local$$20195[topRightInner], local$$20198[left] + local$$20196[3][width], local$$20198[top] + local$$20196[0][width]);
            local$$16510(local$$20209, local$$20197[1], local$$20197[2], local$$20195[topRightInner], local$$20195[bottomRightInner], local$$20198[left] + local$$20198[width] - local$$20196[1][width], local$$20198[top] + local$$20196[0][width]);
            local$$16510(local$$20209, local$$20197[2], local$$20197[3], local$$20195[bottomRightInner], local$$20195[bottomLeftInner], local$$20198[left] + local$$20198[width] - local$$20196[1][width], local$$20198[top] + local$$20198[height] - local$$20196[2][width]);
            local$$16510(local$$20209, local$$20197[3], local$$20197[0], local$$20195[bottomLeftInner], local$$20195[topLeftInner], local$$20198[left] + local$$20196[3][width], local$$20198[top] + local$$20198[height] - local$$20196[2][width]);
            break;
          default:
            local$$16510(local$$20209, local$$20197[0], local$$20197[1], local$$20195[topLeftOuter], local$$20195[topRightOuter], local$$20198[left], local$$20198[top]);
            local$$16510(local$$20209, local$$20197[1], local$$20197[2], local$$20195[topRightOuter], local$$20195[bottomRightOuter], local$$20198[left] + local$$20198[width], local$$20198[top]);
            local$$16510(local$$20209, local$$20197[2], local$$20197[3], local$$20195[bottomRightOuter], local$$20195[bottomLeftOuter], local$$20198[left] + local$$20198[width], local$$20198[top] + local$$20198[height]);
            local$$16510(local$$20209, local$$20197[3], local$$20197[0], local$$20195[bottomLeftOuter], local$$20195[topLeftOuter], local$$20198[left], local$$20198[top] + local$$20198[height]);
            break;
        }
        return local$$20209;
      };
      /** @type {function(?, ?, ?, ?, ?): undefined} */
      local$$14940[exports] = local$$14943;
    }, {
      "./color" : 3,
      "./fontmetrics" : 7,
      "./log" : 13,
      "./nodecontainer" : 14,
      "./pseudoelementcontainer" : 18,
      "./stackingcontext" : 21,
      "./textcontainer" : 25,
      "./utils" : 26,
      "punycode" : 1
    }],
    16 : [function(local$$20474, local$$20475, local$$20476) {
      /**
       * @param {?} local$$20479
       * @param {?} local$$20480
       * @param {?} local$$20481
       * @return {?}
       */
      function local$$20478(local$$20479, local$$20480, local$$20481) {
        /** @type {boolean} */
        var local$$20488 = withCredentials in new XMLHttpRequest;
        if (!local$$20480) {
          return Promise[reject](No proxy configured);
        }
        var local$$20503 = local$$20501(local$$20488);
        var local$$20507 = local$$20505(local$$20480, local$$20479, local$$20503);
        return local$$20488 ? local$$20509(local$$20507) : local$$20511(local$$20481, local$$20507, local$$20503)[then](function(local$$20516) {
          return local$$20518(local$$20516[content]);
        });
      }
      /**
       * @param {?} local$$20532
       * @param {?} local$$20533
       * @param {?} local$$20534
       * @return {?}
       */
      function local$$20531(local$$20532, local$$20533, local$$20534) {
        /** @type {boolean} */
        var local$$20540 = crossOrigin in new Image;
        var local$$20543 = local$$20501(local$$20540);
        var local$$20546 = local$$20505(local$$20533, local$$20532, local$$20543);
        return local$$20540 ? Promise[resolve](local$$20546) : local$$20511(local$$20534, local$$20546, local$$20543)[then](function(local$$20556) {
          return data: + local$$20556[type] + ;base64, + local$$20556[content];
        });
      }
      /**
       * @param {?} local$$20579
       * @param {?} local$$20580
       * @param {?} local$$20581
       * @return {?}
       */
      function local$$20511(local$$20579, local$$20580, local$$20581) {
        return new Promise(function(local$$20583, local$$20584) {
          var local$$20592 = local$$20579[createElement](script);
          /**
           * @return {undefined}
           */
          var local$$20614 = function() {
            delete window[html2canvas][proxy][local$$20581];
            local$$20579[body][removeChild](local$$20592);
          };
          /**
           * @param {?} local$$20623
           * @return {undefined}
           */
          window[html2canvas][proxy][local$$20581] = function(local$$20623) {
            local$$20614();
            local$$20583(local$$20623);
          };
          local$$20592[src] = local$$20580;
          /**
           * @param {?} local$$20641
           * @return {undefined}
           */
          local$$20592[onerror] = function(local$$20641) {
            local$$20614();
            local$$20584(local$$20641);
          };
          local$$20579[body][appendChild](local$$20592);
        });
      }
      /**
       * @param {boolean} local$$20665
       * @return {?}
       */
      function local$$20501(local$$20665) {
        return !local$$20665 ? html2canvas_ + Date[now]() + _ + ++local$$20678 + _ + Math[round](Math[random]() * 1E5) : ;
      }
      /**
       * @param {?} local$$20701
       * @param {?} local$$20702
       * @param {?} local$$20703
       * @return {?}
       */
      function local$$20505(local$$20701, local$$20702, local$$20703) {
        return local$$20701 + ?url= + encodeURIComponent(local$$20702) + (local$$20703[length] ? &callback=html2canvas.proxy. + local$$20703 : );
      }
      /**
       * @param {?} local$$20725
       * @return {?}
       */
      function local$$20724(local$$20725) {
        return function(local$$20727) {
          /** @type {!DOMParser} */
          var local$$20731 = new DOMParser;
          var local$$20733;
          try {
            local$$20733 = local$$20731[parseFromString](local$$20727, text/html);
          } catch (local$$20744) {
            local$$20745(DOMParser not supported, falling back to createHTMLDocument);
            local$$20733 = document[implementation][createHTMLDocument]();
            try {
              local$$20733[open]();
              local$$20733[write](local$$20727);
              local$$20733[close]();
            } catch (local$$20777) {
              local$$20745(createHTMLDocument write not supported, falling back to document.body.innerHTML);
              local$$20733[body][innerHTML] = local$$20727;
            }
          }
          var local$$20805 = local$$20733[querySelector](base);
          if (!local$$20805 || !local$$20805[href][host]) {
            var local$$20822 = local$$20733[createElement](base);
            local$$20822[href] = local$$20725;
            local$$20733[head][insertBefore](local$$20822, local$$20733[head][firstChild]);
          }
          return local$$20733;
        };
      }
      /**
       * @param {?} local$$20853
       * @param {boolean} local$$20854
       * @param {?} local$$20855
       * @param {?} local$$20856
       * @param {?} local$$20857
       * @param {?} local$$20858
       * @return {?}
       */
      function local$$20852(local$$20853, local$$20854, local$$20855, local$$20856, local$$20857, local$$20858) {
        return (new local$$20478(local$$20853, local$$20854, window[document]))[then](local$$20724(local$$20853))[then](function(local$$20872) {
          return local$$20874(local$$20872, local$$20855, local$$20856, local$$20857, local$$20858, 0, 0);
        });
      }
      var local$$20509 = local$$20474(./xhr);
      var local$$20892 = local$$20474(./utils);
      var local$$20745 = local$$20474(./log);
      var local$$20874 = local$$20474(./clone);
      var local$$20518 = local$$20892[decode64];
      /** @type {number} */
      var local$$20678 = 0;
      /** @type {function(?, ?, ?): ?} */
      local$$20476[Proxy] = local$$20478;
      /** @type {function(?, ?, ?): ?} */
      local$$20476[ProxyURL] = local$$20531;
      /** @type {function(?, boolean, ?, ?, ?, ?): ?} */
      local$$20476[loadUrlDocument] = local$$20852;
    }, {
      "./clone" : 2,
      "./log" : 13,
      "./utils" : 26,
      "./xhr" : 28
    }],
    17 : [function(local$$20931, local$$20932, local$$20933) {
      /**
       * @param {?} local$$20936
       * @param {!Object} local$$20937
       * @return {undefined}
       */
      function local$$20935(local$$20936, local$$20937) {
        var local$$20945 = document[createElement](a);
        local$$20945[href] = local$$20936;
        local$$20936 = local$$20945[href];
        this[src] = local$$20936;
        /** @type {!Image} */
        this[image] = new Image;
        var local$$20968 = this;
        /** @type {!Promise} */
        this[promise] = new Promise(function(local$$20973, local$$20974) {
          local$$20968[image][crossOrigin] = Anonymous;
          local$$20968[image][onload] = local$$20973;
          local$$20968[image][onerror] = local$$20974;
          (new local$$21002(local$$20936, local$$20937, document))[then](function(local$$21007) {
            local$$20968[image][src] = local$$21007;
          })[catch](local$$20974);
        });
      }
      var local$$21002 = local$$20931(./proxy)[ProxyURL];
      /** @type {function(?, !Object): undefined} */
      local$$20932[exports] = local$$20935;
    }, {
      "./proxy" : 16
    }],
    18 : [function(local$$21050, local$$21051, local$$21052) {
      /**
       * @param {?} local$$21055
       * @param {?} local$$21056
       * @param {?} local$$21057
       * @return {undefined}
       */
      function local$$21054(local$$21055, local$$21056, local$$21057) {
        local$$21059[call](this, local$$21055, local$$21056);
        /** @type {boolean} */
        this[isPseudoElement] = true;
        /** @type {boolean} */
        this[before] = local$$21057 === :before;
      }
      var local$$21059 = local$$21050(./nodecontainer);
      /**
       * @param {?} local$$21091
       * @return {undefined}
       */
      local$$21054[prototype][cloneTo] = function(local$$21091) {
        local$$21054[prototype][cloneTo][call](this, local$$21091);
        /** @type {boolean} */
        local$$21091[isPseudoElement] = true;
        local$$21091[before] = this[before];
      };
      local$$21054[prototype] = Object[create](local$$21059[prototype]);
      /**
       * @return {undefined}
       */
      local$$21054[prototype][appendToDOM] = function() {
        if (this[before]) {
          this[parent][node][insertBefore](this[node], this[parent][node][firstChild]);
        } else {
          this[parent][node][appendChild](this[node]);
        }
        this[parent][node][className] +=   + this[getHideClass]();
      };
      /**
       * @return {undefined}
       */
      local$$21054[prototype][cleanDOM] = function() {
        this[node][parentNode][removeChild](this[node]);
        this[parent][node][className] = this[parent][node][className][replace](this[getHideClass](), );
      };
      /**
       * @return {?}
       */
      local$$21054[prototype][getHideClass] = function() {
        return this[PSEUDO_HIDE_ELEMENT_CLASS_ + (this[before] ? BEFORE : AFTER)];
      };
      local$$21054[prototype][PSEUDO_HIDE_ELEMENT_CLASS_BEFORE] = ___html2canvas___pseudoelement_before;
      local$$21054[prototype][PSEUDO_HIDE_ELEMENT_CLASS_AFTER] = ___html2canvas___pseudoelement_after;
      /** @type {function(?, ?, ?): undefined} */
      local$$21051[exports] = local$$21054;
    }, {
      "./nodecontainer" : 14
    }],
    19 : [function(local$$21317, local$$21318, local$$21319) {
      /**
       * @param {?} local$$21322
       * @param {?} local$$21323
       * @param {?} local$$21324
       * @param {?} local$$21325
       * @param {?} local$$21326
       * @return {undefined}
       */
      function local$$21321(local$$21322, local$$21323, local$$21324, local$$21325, local$$21326) {
        this[width] = local$$21322;
        this[height] = local$$21323;
        this[images] = local$$21324;
        this[options] = local$$21325;
        this[document] = local$$21326;
      }
      var local$$21358 = local$$21317(./log);
      /**
       * @param {?} local$$21366
       * @param {?} local$$21367
       * @param {?} local$$21368
       * @param {?} local$$21369
       * @return {undefined}
       */
      local$$21321[prototype][renderImage] = function(local$$21366, local$$21367, local$$21368, local$$21369) {
        var local$$21377 = local$$21366[cssInt](paddingLeft);
        var local$$21385 = local$$21366[cssInt](paddingTop);
        var local$$21393 = local$$21366[cssInt](paddingRight);
        var local$$21401 = local$$21366[cssInt](paddingBottom);
        var local$$21406 = local$$21368[borders];
        /** @type {number} */
        var local$$21425 = local$$21367[width] - (local$$21406[1][width] + local$$21406[3][width] + local$$21377 + local$$21393);
        /** @type {number} */
        var local$$21444 = local$$21367[height] - (local$$21406[0][width] + local$$21406[2][width] + local$$21385 + local$$21401);
        this[drawImage](local$$21369, 0, 0, local$$21369[image][width] || local$$21425, local$$21369[image][height] || local$$21444, local$$21367[left] + local$$21377 + local$$21406[3][width], local$$21367[top] + local$$21385 + local$$21406[0][width], local$$21425, local$$21444);
      };
      /**
       * @param {?} local$$21497
       * @param {?} local$$21498
       * @param {?} local$$21499
       * @return {undefined}
       */
      local$$21321[prototype][renderBackground] = function(local$$21497, local$$21498, local$$21499) {
        if (local$$21498[height] > 0 && local$$21498[width] > 0) {
          this[renderBackgroundColor](local$$21497, local$$21498);
          this[renderBackgroundImage](local$$21497, local$$21498, local$$21499);
        }
      };
      /**
       * @param {?} local$$21534
       * @param {?} local$$21535
       * @return {undefined}
       */
      local$$21321[prototype][renderBackgroundColor] = function(local$$21534, local$$21535) {
        var local$$21543 = local$$21534[color](backgroundColor);
        if (!local$$21543[isTransparent]()) {
          this[rectangle](local$$21535[left], local$$21535[top], local$$21535[width], local$$21535[height], local$$21543);
        }
      };
      /**
       * @param {?} local$$21579
       * @return {undefined}
       */
      local$$21321[prototype][renderBorders] = function(local$$21579) {
        local$$21579[forEach](this[renderBorder], this);
      };
      /**
       * @param {?} local$$21599
       * @return {undefined}
       */
      local$$21321[prototype][renderBorder] = function(local$$21599) {
        if (!local$$21599[color][isTransparent]() && local$$21599[args] !== null) {
          this[drawShape](local$$21599[args], local$$21599[color]);
        }
      };
      /**
       * @param {?} local$$21638
       * @param {?} local$$21639
       * @param {?} local$$21640
       * @return {undefined}
       */
      local$$21321[prototype][renderBackgroundImage] = function(local$$21638, local$$21639, local$$21640) {
        var local$$21646 = local$$21638[parseBackgroundImages]();
        local$$21646[reverse]()[forEach](function(local$$21655, local$$21656, local$$21657) {
          switch(local$$21655[method]) {
            case url:
              var local$$21676 = this[images][get](local$$21655[args][0]);
              if (local$$21676) {
                this[renderBackgroundRepeating](local$$21638, local$$21639, local$$21676, local$$21657[length] - (local$$21656 + 1), local$$21640);
              } else {
                local$$21358(Error loading background-image, local$$21655[args][0]);
              }
              break;
            case linear-gradient:
            case gradient:
              var local$$21722 = this[images][get](local$$21655[value]);
              if (local$$21722) {
                this[renderBackgroundGradient](local$$21722, local$$21639, local$$21640);
              } else {
                local$$21358(Error loading background-image, local$$21655[args][0]);
              }
              break;
            case none:
              break;
            default:
              local$$21358(Unknown background-image type, local$$21655[args][0]);
          }
        }, this);
      };
      /**
       * @param {?} local$$21776
       * @param {?} local$$21777
       * @param {?} local$$21778
       * @param {?} local$$21779
       * @param {!Array} local$$21780
       * @return {undefined}
       */
      local$$21321[prototype][renderBackgroundRepeating] = function(local$$21776, local$$21777, local$$21778, local$$21779, local$$21780) {
        var local$$21789 = local$$21776[parseBackgroundSize](local$$21777, local$$21778[image], local$$21779);
        var local$$21798 = local$$21776[parseBackgroundPosition](local$$21777, local$$21778[image], local$$21779, local$$21789);
        var local$$21804 = local$$21776[parseBackgroundRepeat](local$$21779);
        switch(local$$21804) {
          case repeat-x:
          case repeat no-repeat:
            this[backgroundRepeatShape](local$$21778, local$$21798, local$$21789, local$$21777, local$$21777[left] + local$$21780[3], local$$21777[top] + local$$21798[top] + local$$21780[0], 99999, local$$21789[height], local$$21780);
            break;
          case repeat-y:
          case no-repeat repeat:
            this[backgroundRepeatShape](local$$21778, local$$21798, local$$21789, local$$21777, local$$21777[left] + local$$21798[left] + local$$21780[3], local$$21777[top] + local$$21780[0], local$$21789[width], 99999, local$$21780);
            break;
          case no-repeat:
            this[backgroundRepeatShape](local$$21778, local$$21798, local$$21789, local$$21777, local$$21777[left] + local$$21798[left] + local$$21780[3], local$$21777[top] + local$$21798[top] + local$$21780[0], local$$21789[width], local$$21789[height], local$$21780);
            break;
          default:
            this[renderBackgroundRepeat](local$$21778, local$$21798, local$$21789, {
              top : local$$21777[top],
              left : local$$21777[left]
            }, local$$21780[3], local$$21780[0]);
            break;
        }
      };
      /** @type {function(?, ?, ?, ?, ?): undefined} */
      local$$21318[exports] = local$$21321;
    }, {
      "./log" : 13
    }],
    20 : [function(local$$21947, local$$21948, local$$21949) {
      /**
       * @param {?} local$$21952
       * @param {?} local$$21953
       * @return {undefined}
       */
      function local$$21951(local$$21952, local$$21953) {
        local$$21955[apply](this, arguments);
        this[canvas] = this[options][canvas] || this[document][createElement](canvas);
        if (!this[options][canvas]) {
          this[canvas][width] = local$$21952;
          this[canvas][height] = local$$21953;
        }
        this[ctx] = this[canvas][getContext](2d);
        this[taintCtx] = this[document][createElement](canvas)[getContext](2d);
        this[ctx][textBaseline] = bottom;
        this[variables] = {};
        local$$22058(Initialized CanvasRenderer with size, local$$21952, x, local$$21953);
      }
      /**
       * @param {?} local$$22068
       * @return {?}
       */
      function local$$22067(local$$22068) {
        return local$$22068[length] > 0;
      }
      var local$$21955 = local$$21947(../renderer);
      var local$$22085 = local$$21947(../lineargradientcontainer);
      var local$$22058 = local$$21947(../log);
      local$$21951[prototype] = Object[create](local$$21955[prototype]);
      /**
       * @param {string} local$$22109
       * @return {?}
       */
      local$$21951[prototype][setFillStyle] = function(local$$22109) {
        this[ctx][fillStyle] = typeof local$$22109 === object && !!local$$22109[isColor] ? local$$22109.toString() : local$$22109;
        return this[ctx];
      };
      /**
       * @param {?} local$$22146
       * @param {?} local$$22147
       * @param {?} local$$22148
       * @param {?} local$$22149
       * @param {?} local$$22150
       * @return {undefined}
       */
      local$$21951[prototype][rectangle] = function(local$$22146, local$$22147, local$$22148, local$$22149, local$$22150) {
        this[setFillStyle](local$$22150)[fillRect](local$$22146, local$$22147, local$$22148, local$$22149);
      };
      /**
       * @param {number} local$$22171
       * @param {number} local$$22172
       * @param {number} local$$22173
       * @param {?} local$$22174
       * @return {undefined}
       */
      local$$21951[prototype][circle] = function(local$$22171, local$$22172, local$$22173, local$$22174) {
        this[setFillStyle](local$$22174);
        this[ctx][beginPath]();
        this[ctx][arc](local$$22171 + local$$22173 / 2, local$$22172 + local$$22173 / 2, local$$22173 / 2, 0, Math[PI] * 2, true);
        this[ctx][closePath]();
        this[ctx][fill]();
      };
      /**
       * @param {?} local$$22238
       * @param {?} local$$22239
       * @param {?} local$$22240
       * @param {?} local$$22241
       * @param {?} local$$22242
       * @param {string} local$$22243
       * @return {undefined}
       */
      local$$21951[prototype][circleStroke] = function(local$$22238, local$$22239, local$$22240, local$$22241, local$$22242, local$$22243) {
        this[circle](local$$22238, local$$22239, local$$22240, local$$22241);
        this[ctx][strokeStyle] = local$$22243.toString();
        this[ctx][stroke]();
      };
      /**
       * @param {?} local$$22278
       * @param {?} local$$22279
       * @return {undefined}
       */
      local$$21951[prototype][drawShape] = function(local$$22278, local$$22279) {
        this[shape](local$$22278);
        this[setFillStyle](local$$22279)[fill]();
      };
      /**
       * @param {?} local$$22305
       * @return {?}
       */
      local$$21951[prototype][taints] = function(local$$22305) {
        if (local$$22305[tainted] === null) {
          this[taintCtx][drawImage](local$$22305[image], 0, 0);
          try {
            this[taintCtx][getImageData](0, 0, 1, 1);
            /** @type {boolean} */
            local$$22305[tainted] = false;
          } catch (local$$22344) {
            this[taintCtx] = document[createElement](canvas)[getContext](2d);
            /** @type {boolean} */
            local$$22305[tainted] = true;
          }
        }
        return local$$22305[tainted];
      };
      /**
       * @param {?} local$$22389
       * @param {?} local$$22390
       * @param {?} local$$22391
       * @param {?} local$$22392
       * @param {?} local$$22393
       * @param {?} local$$22394
       * @param {?} local$$22395
       * @param {?} local$$22396
       * @param {?} local$$22397
       * @return {undefined}
       */
      local$$21951[prototype][drawImage] = function(local$$22389, local$$22390, local$$22391, local$$22392, local$$22393, local$$22394, local$$22395, local$$22396, local$$22397) {
        if (!this[taints](local$$22389) || this[options][allowTaint]) {
          this[ctx][drawImage](local$$22389[image], local$$22390, local$$22391, local$$22392, local$$22393, local$$22394, local$$22395, local$$22396, local$$22397);
        }
      };
      /**
       * @param {?} local$$22434
       * @param {?} local$$22435
       * @param {?} local$$22436
       * @return {undefined}
       */
      local$$21951[prototype][clip] = function(local$$22434, local$$22435, local$$22436) {
        this[ctx][save]();
        local$$22434[filter](local$$22067)[forEach](function(local$$22453) {
          this[shape](local$$22453)[clip]();
        }, this);
        local$$22435[call](local$$22436);
        this[ctx][restore]();
      };
      /**
       * @param {?} local$$22491
       * @return {?}
       */
      local$$21951[prototype][shape] = function(local$$22491) {
        this[ctx][beginPath]();
        local$$22491[forEach](function(local$$22504, local$$22505) {
          if (local$$22504[0] === rect) {
            this[ctx][rect][apply](this[ctx], local$$22504[slice](1));
          } else {
            this[ctx][local$$22505 === 0 ? moveTo : local$$22504[0] + To][apply](this[ctx], local$$22504[slice](1));
          }
        }, this);
        this[ctx][closePath]();
        return this[ctx];
      };
      /**
       * @param {?} local$$22587
       * @param {?} local$$22588
       * @param {?} local$$22589
       * @param {?} local$$22590
       * @param {?} local$$22591
       * @param {?} local$$22592
       * @return {undefined}
       */
      local$$21951[prototype][font] = function(local$$22587, local$$22588, local$$22589, local$$22590, local$$22591, local$$22592) {
        this[setFillStyle](local$$22587)[font] = [local$$22588, local$$22589, local$$22590, local$$22591, local$$22592][join]( )[split](,)[0];
      };
      /**
       * @param {string} local$$22628
       * @param {?} local$$22629
       * @param {?} local$$22630
       * @param {?} local$$22631
       * @return {undefined}
       */
      local$$21951[prototype][fontShadow] = function(local$$22628, local$$22629, local$$22630, local$$22631) {
        this[setVariable](shadowColor, local$$22628.toString())[setVariable](shadowOffsetY, local$$22629)[setVariable](shadowOffsetX, local$$22630)[setVariable](shadowBlur, local$$22631);
      };
      /**
       * @return {undefined}
       */
      local$$21951[prototype][clearShadow] = function() {
        this[setVariable](shadowColor, rgba(0,0,0,0));
      };
      /**
       * @param {?} local$$22690
       * @return {undefined}
       */
      local$$21951[prototype][setOpacity] = function(local$$22690) {
        this[ctx][globalAlpha] = local$$22690;
      };
      /**
       * @param {?} local$$22710
       * @return {undefined}
       */
      local$$21951[prototype][setTransform] = function(local$$22710) {
        this[ctx][translate](local$$22710[origin][0], local$$22710[origin][1]);
        this[ctx][transform][apply](this[ctx], local$$22710[matrix]);
        this[ctx][translate](-local$$22710[origin][0], -local$$22710[origin][1]);
      };
      /**
       * @param {?} local$$22777
       * @param {?} local$$22778
       * @return {?}
       */
      local$$21951[prototype][setVariable] = function(local$$22777, local$$22778) {
        if (this[variables][local$$22777] !== local$$22778) {
          this[variables][local$$22777] = this[ctx][local$$22777] = local$$22778;
        }
        return this;
      };
      /**
       * @param {?} local$$22810
       * @param {?} local$$22811
       * @param {?} local$$22812
       * @return {undefined}
       */
      local$$21951[prototype][text] = function(local$$22810, local$$22811, local$$22812) {
        this[ctx][fillText](local$$22810, local$$22811, local$$22812);
      };
      /**
       * @param {?} local$$22832
       * @param {?} local$$22833
       * @param {?} local$$22834
       * @param {?} local$$22835
       * @param {(Object|number)} local$$22836
       * @param {!Object} local$$22837
       * @param {!Object} local$$22838
       * @param {(Object|number)} local$$22839
       * @param {!Array} local$$22840
       * @return {undefined}
       */
      local$$21951[prototype][backgroundRepeatShape] = function(local$$22832, local$$22833, local$$22834, local$$22835, local$$22836, local$$22837, local$$22838, local$$22839, local$$22840) {
        /** @type {!Array} */
        var local$$22891 = [[line, Math[round](local$$22836), Math[round](local$$22837)], [line, Math[round](local$$22836 + local$$22838), Math[round](local$$22837)], [line, Math[round](local$$22836 + local$$22838), Math[round](local$$22839 + local$$22837)], [line, Math[round](local$$22836), Math[round](local$$22839 + local$$22837)]];
        this[clip]([local$$22891], function() {
          this[renderBackgroundRepeat](local$$22832, local$$22833, local$$22834, local$$22835, local$$22840[3], local$$22840[0]);
        }, this);
      };
      /**
       * @param {?} local$$22921
       * @param {?} local$$22922
       * @param {?} local$$22923
       * @param {?} local$$22924
       * @param {?} local$$22925
       * @param {?} local$$22926
       * @return {undefined}
       */
      local$$21951[prototype][renderBackgroundRepeat] = function(local$$22921, local$$22922, local$$22923, local$$22924, local$$22925, local$$22926) {
        var local$$22940 = Math[round](local$$22924[left] + local$$22922[left] + local$$22925);
        var local$$22954 = Math[round](local$$22924[top] + local$$22922[top] + local$$22926);
        this[setFillStyle](this[ctx][_0x34b6[1E3]](this[resizeImage](local$$22921, local$$22923), repeat));
        this[ctx][translate](local$$22940, local$$22954);
        this[ctx][fill]();
        this[ctx][translate](-local$$22940, -local$$22954);
      };
      /**
       * @param {?} local$$23010
       * @param {?} local$$23011
       * @return {undefined}
       */
      local$$21951[prototype][renderBackgroundGradient] = function(local$$23010, local$$23011) {
        if (local$$23010 instanceof local$$22085) {
          var local$$23065 = this[ctx][createLinearGradient](local$$23011[left] + local$$23011[width] * local$$23010[x0], local$$23011[top] + local$$23011[height] * local$$23010[y0], local$$23011[left] + local$$23011[width] * local$$23010[x1], local$$23011[top] + local$$23011[height] * local$$23010[y1]);
          local$$23010[colorStops][forEach](function(local$$23073) {
            local$$23065[addColorStop](local$$23073[stop], local$$23073[color].toString());
          });
          this[rectangle](local$$23011[left], local$$23011[top], local$$23011[width], local$$23011[height], local$$23065);
        }
      };
      /**
       * @param {?} local$$23121
       * @param {?} local$$23122
       * @return {?}
       */
      local$$21951[prototype][resizeImage] = function(local$$23121, local$$23122) {
        var local$$23127 = local$$23121[image];
        if (local$$23127[width] === local$$23122[width] && local$$23127[height] === local$$23122[height]) {
          return local$$23127;
        }
        var local$$23148;
        var local$$23156 = document[createElement](canvas);
        local$$23156[width] = local$$23122[width];
        local$$23156[height] = local$$23122[height];
        local$$23148 = local$$23156[getContext](2d);
        local$$23148[drawImage](local$$23127, 0, 0, local$$23127[width], local$$23127[height], 0, 0, local$$23122[width], local$$23122[height]);
        return local$$23156;
      };
      /** @type {function(?, ?): undefined} */
      local$$21948[exports] = local$$21951;
    }, {
      "../lineargradientcontainer" : 12,
      "../log" : 13,
      "../renderer" : 19
    }],
    21 : [function(local$$23221, local$$23222, local$$23223) {
      /**
       * @param {?} local$$23226
       * @param {?} local$$23227
       * @param {?} local$$23228
       * @param {?} local$$23229
       * @return {undefined}
       */
      function local$$23225(local$$23226, local$$23227, local$$23228, local$$23229) {
        local$$23231[call](this, local$$23228, local$$23229);
        this[ownStacking] = local$$23226;
        /** @type {!Array} */
        this[contexts] = [];
        /** @type {!Array} */
        this[children] = [];
        /** @type {number} */
        this[opacity] = (this[parent] ? this[parent][stack][opacity] : 1) * local$$23227;
      }
      var local$$23231 = local$$23221(./nodecontainer);
      local$$23225[prototype] = Object[create](local$$23231[prototype]);
      /**
       * @param {?} local$$23298
       * @return {?}
       */
      local$$23225[prototype][getParentStack] = function(local$$23298) {
        var local$$23311 = this[parent] ? this[parent][stack] : null;
        return local$$23311 ? local$$23311[ownStacking] ? local$$23311 : local$$23311[getParentStack](local$$23298) : local$$23298[stack];
      };
      /** @type {function(?, ?, ?, ?): undefined} */
      local$$23222[exports] = local$$23225;
    }, {
      "./nodecontainer" : 14
    }],
    22 : [function(local$$23341, local$$23342, local$$23343) {
      /**
       * @param {?} local$$23346
       * @return {undefined}
       */
      function local$$23345(local$$23346) {
        this[rangeBounds] = this[testRangeBounds](local$$23346);
        this[cors] = this[testCORS]();
        this[svg] = this[testSVG]();
      }
      /**
       * @param {?} local$$23383
       * @return {?}
       */
      local$$23345[prototype][testRangeBounds] = function(local$$23383) {
        var local$$23385;
        var local$$23387;
        var local$$23389;
        var local$$23391;
        /** @type {boolean} */
        var local$$23394 = false;
        if (local$$23383[createRange]) {
          local$$23385 = local$$23383[createRange]();
          if (local$$23385[getBoundingClientRect]) {
            local$$23387 = local$$23383[createElement](boundtest);
            local$$23387[style][height] = 123px;
            local$$23387[style][display] = block;
            local$$23383[body][appendChild](local$$23387);
            local$$23385[selectNode](local$$23387);
            local$$23389 = local$$23385[getBoundingClientRect]();
            local$$23391 = local$$23389[height];
            if (local$$23391 === 123) {
              /** @type {boolean} */
              local$$23394 = true;
            }
            local$$23383[body][removeChild](local$$23387);
          }
        }
        return local$$23394;
      };
      /**
       * @return {?}
       */
      local$$23345[prototype][testCORS] = function() {
        return typeof(new Image)[crossOrigin] !== undefined;
      };
      /**
       * @return {?}
       */
      local$$23345[prototype][testSVG] = function() {
        /** @type {!Image} */
        var local$$23514 = new Image;
        var local$$23522 = document[createElement](canvas);
        var local$$23530 = local$$23522[getContext](2d);
        local$$23514[src] = data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'></svg>;
        try {
          local$$23530[drawImage](local$$23514, 0, 0);
          local$$23522[toDataURL]();
        } catch (local$$23552) {
          return false;
        }
        return true;
      };
      /** @type {function(?): undefined} */
      local$$23342[exports] = local$$23345;
    }, {}],
    23 : [function(local$$23576, local$$23577, local$$23578) {
      /**
       * @param {?} local$$23581
       * @return {undefined}
       */
      function local$$23580(local$$23581) {
        this[src] = local$$23581;
        /** @type {null} */
        this[image] = null;
        var local$$23594 = this;
        this[promise] = this[hasFabric]()[then](function() {
          return local$$23594[isInline](local$$23581) ? Promise[resolve](local$$23594[inlineFormatting](local$$23581)) : local$$23619(local$$23581);
        })[then](function(local$$23629) {
          return new Promise(function(local$$23631) {
            window[html2canvas][svg][fabric][loadSVGFromString](local$$23629, local$$23594[createCanvas][call](local$$23594, local$$23631));
          });
        });
      }
      var local$$23619 = local$$23576(./xhr);
      var local$$23675 = local$$23576(./utils)[decode64];
      /**
       * @return {?}
       */
      local$$23580[prototype][hasFabric] = function() {
        return !window[html2canvas][svg] || !window[html2canvas][svg][fabric] ? Promise[reject](new Error(html2canvas.svg.js is not loaded, cannot render svg)) : Promise[resolve]();
      };
      /**
       * @param {?} local$$23725
       * @return {?}
       */
      local$$23580[prototype][inlineFormatting] = function(local$$23725) {
        return /^data:image\/svg\+xml;base64,/[test](local$$23725) ? this[decode64](this[removeContentType](local$$23725)) : this[removeContentType](local$$23725);
      };
      /**
       * @param {?} local$$23757
       * @return {?}
       */
      local$$23580[prototype][removeContentType] = function(local$$23757) {
        return local$$23757[replace](/^data:image\/svg\+xml(;base64)?,/, );
      };
      /**
       * @param {?} local$$23778
       * @return {?}
       */
      local$$23580[prototype][isInline] = function(local$$23778) {
        return /^data:image\/svg\+xml/i[test](local$$23778);
      };
      /**
       * @param {?} local$$23797
       * @return {?}
       */
      local$$23580[prototype][createCanvas] = function(local$$23797) {
        var local$$23799 = this;
        return function(local$$23801, local$$23802) {
          var local$$23818 = new window[html2canvas][svg][fabric].StaticCanvas(c);
          local$$23799[image] = local$$23818[lowerCanvasEl];
          local$$23818[setWidth](local$$23802[width])[setHeight](local$$23802[height])[add](window[html2canvas][svg][fabric][util][groupSVGElements](local$$23801, local$$23802))[renderAll]();
          local$$23797(local$$23818[lowerCanvasEl]);
        };
      };
      /**
       * @param {?} local$$23885
       * @return {?}
       */
      local$$23580[prototype][decode64] = function(local$$23885) {
        return typeof window[atob] === function ? window[atob](local$$23885) : local$$23675(local$$23885);
      };
      /** @type {function(?): undefined} */
      local$$23577[exports] = local$$23580;
    }, {
      "./utils" : 26,
      "./xhr" : 28
    }],
    24 : [function(local$$23917, local$$23918, local$$23919) {
      /**
       * @param {?} local$$23922
       * @param {!Object} local$$23923
       * @return {undefined}
       */
      function local$$23921(local$$23922, local$$23923) {
        this[src] = local$$23922;
        /** @type {null} */
        this[image] = null;
        var local$$23936 = this;
        this[promise] = local$$23923 ? new Promise(function(local$$23941, local$$23942) {
          /** @type {!Image} */
          local$$23936[image] = new Image;
          local$$23936[image][onload] = local$$23941;
          local$$23936[image][onerror] = local$$23942;
          local$$23936[image][src] = data:image/svg+xml, + (new XMLSerializer)[serializeToString](local$$23922);
          if (local$$23936[image][complete] === true) {
            local$$23941(local$$23936[image]);
          }
        }) : this[hasFabric]()[then](function() {
          return new Promise(function(local$$24009) {
            window[html2canvas][svg][fabric][parseSVGDocument](local$$23922, local$$23936[createCanvas][call](local$$23936, local$$24009));
          });
        });
      }
      var local$$24047 = local$$23917(./svgcontainer);
      local$$23921[prototype] = Object[create](local$$24047[prototype]);
      /** @type {function(?, !Object): undefined} */
      local$$23918[exports] = local$$23921;
    }, {
      "./svgcontainer" : 23
    }],
    25 : [function(local$$24072, local$$24073, local$$24074) {
      /**
       * @param {?} local$$24077
       * @param {?} local$$24078
       * @return {undefined}
       */
      function local$$24076(local$$24077, local$$24078) {
        local$$24080[call](this, local$$24077, local$$24078);
      }
      /**
       * @param {?} local$$24089
       * @param {?} local$$24090
       * @param {?} local$$24091
       * @return {?}
       */
      function local$$24088(local$$24089, local$$24090, local$$24091) {
        if (local$$24089[length] > 0) {
          return local$$24090 + local$$24091[toUpperCase]();
        }
      }
      var local$$24080 = local$$24072(./nodecontainer);
      local$$24076[prototype] = Object[create](local$$24080[prototype]);
      /**
       * @return {undefined}
       */
      local$$24076[prototype][applyTextTransform] = function() {
        this[node][data] = this[transform](this[parent][css](textTransform));
      };
      /**
       * @param {?} local$$24162
       * @return {?}
       */
      local$$24076[prototype][transform] = function(local$$24162) {
        var local$$24170 = this[node][data];
        switch(local$$24162) {
          case lowercase:
            return local$$24170[toLowerCase]();
          case capitalize:
            return local$$24170[replace](/(^|\s|:|-|\(|\))([a-z])/g, local$$24088);
          case uppercase:
            return local$$24170[toUpperCase]();
          default:
            return local$$24170;
        }
      };
      /** @type {function(?, ?): undefined} */
      local$$24073[exports] = local$$24076;
    }, {
      "./nodecontainer" : 14
    }],
    26 : [function(local$$24220, local$$24221, local$$24222) {
      /**
       * @return {?}
       */
      local$$24222[smallImage] = function local$$24227() {
        return data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7;
      };
      /**
       * @param {?} local$$24239
       * @param {?} local$$24240
       * @return {?}
       */
      local$$24222[bind] = function(local$$24239, local$$24240) {
        return function() {
          return local$$24239[apply](local$$24240, arguments);
        };
      };
      /**
       * @param {!Object} local$$24258
       * @return {?}
       */
      local$$24222[decode64] = function(local$$24258) {
        var local$$24262 = ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/;
        var local$$24267 = local$$24258[length];
        var local$$24269;
        var local$$24271;
        var local$$24273;
        var local$$24275;
        var local$$24277;
        var local$$24279;
        var local$$24281;
        var local$$24283;
        var local$$24287 = ;
        /** @type {number} */
        local$$24269 = 0;
        for (; local$$24269 < local$$24267; local$$24269 = local$$24269 + 4) {
          local$$24271 = local$$24262[indexOf](local$$24258[local$$24269]);
          local$$24273 = local$$24262[indexOf](local$$24258[local$$24269 + 1]);
          local$$24275 = local$$24262[indexOf](local$$24258[local$$24269 + 2]);
          local$$24277 = local$$24262[indexOf](local$$24258[local$$24269 + 3]);
          /** @type {number} */
          local$$24279 = local$$24271 << 2 | local$$24273 >> 4;
          /** @type {number} */
          local$$24281 = (local$$24273 & 15) << 4 | local$$24275 >> 2;
          /** @type {number} */
          local$$24283 = (local$$24275 & 3) << 6 | local$$24277;
          if (local$$24275 === 64) {
            local$$24287 = local$$24287 + String[fromCharCode](local$$24279);
          } else {
            if (local$$24277 === 64 || local$$24277 === -1) {
              local$$24287 = local$$24287 + String[fromCharCode](local$$24279, local$$24281);
            } else {
              local$$24287 = local$$24287 + String[fromCharCode](local$$24279, local$$24281, local$$24283);
            }
          }
        }
        return local$$24287;
      };
      /**
       * @param {?} local$$24399
       * @return {?}
       */
      local$$24222[getBounds] = function(local$$24399) {
        if (local$$24399[getBoundingClientRect]) {
          var local$$24408 = local$$24399[getBoundingClientRect]();
          var local$$24422 = local$$24399[offsetWidth] == null ? local$$24408[width] : local$$24399[offsetWidth];
          return {
            top : local$$24408[top],
            bottom : local$$24408[bottom] || local$$24408[top] + local$$24408[height],
            right : local$$24408[left] + local$$24422,
            left : local$$24408[left],
            width : local$$24422,
            height : local$$24399[offsetHeight] == null ? local$$24408[height] : local$$24399[offsetHeight]
          };
        }
        return {};
      };
      /**
       * @param {?} local$$24471
       * @return {?}
       */
      local$$24222[offsetBounds] = function(local$$24471) {
        var local$$24487 = local$$24471[offsetParent] ? local$$24222[offsetBounds](local$$24471[offsetParent]) : {
          top : 0,
          left : 0
        };
        return {
          top : local$$24471[offsetTop] + local$$24487[top],
          bottom : local$$24471[offsetTop] + local$$24471[offsetHeight] + local$$24487[top],
          right : local$$24471[offsetLeft] + local$$24487[left] + local$$24471[offsetWidth],
          left : local$$24471[offsetLeft] + local$$24487[left],
          width : local$$24471[offsetWidth],
          height : local$$24471[offsetHeight]
        };
      };
      /**
       * @param {?} local$$24540
       * @return {?}
       */
      local$$24222[parseBackgrounds] = function(local$$24540) {
        var local$$24544 =  
	;
        var local$$24546;
        var local$$24548;
        var local$$24550;
        var local$$24552;
        var local$$24554;
        /** @type {!Array} */
        var local$$24557 = [];
        /** @type {number} */
        var local$$24560 = 0;
        /** @type {number} */
        var local$$24563 = 0;
        var local$$24565;
        var local$$24567;
        /**
         * @return {undefined}
         */
        var local$$24667 = function() {
          if (local$$24546) {
            if (local$$24548[substr](0, 1) === ") {
              local$$24548 = local$$24548[substr](1, local$$24548[length] - 2);
            }
            if (local$$24548) {
              local$$24567[push](local$$24548);
            }
            if (local$$24546[substr](0, 1) === - && (local$$24552 = local$$24546[indexOf](-, 1) + 1) > 0) {
              local$$24550 = local$$24546[substr](0, local$$24552);
              local$$24546 = local$$24546[substr](local$$24552);
            }
            local$$24557[push]({
              prefix : local$$24550,
              method : local$$24546[toLowerCase](),
              value : local$$24554,
              args : local$$24567,
              image : null
            });
          }
          /** @type {!Array} */
          local$$24567 = [];
          local$$24546 = local$$24550 = local$$24548 = local$$24554 = ;
        };
        /** @type {!Array} */
        local$$24567 = [];
        local$$24546 = local$$24550 = local$$24548 = local$$24554 = ;
        local$$24540[split]()[forEach](function(local$$24688) {
          if (local$$24560 === 0 && local$$24544[indexOf](local$$24688) > -1) {
            return;
          }
          switch(local$$24688) {
            case ":
              if (!local$$24565) {
                /** @type {!AudioNode} */
                local$$24565 = local$$24688;
              } else {
                if (local$$24565 === local$$24688) {
                  /** @type {null} */
                  local$$24565 = null;
                }
              }
              break;
            case (:
              if (local$$24565) {
                break;
              } else {
                if (local$$24560 === 0) {
                  /** @type {number} */
                  local$$24560 = 1;
                  local$$24554 = local$$24554 + local$$24688;
                  return;
                } else {
                  local$$24563++;
                }
              }
              break;
            case ):
              if (local$$24565) {
                break;
              } else {
                if (local$$24560 === 1) {
                  if (local$$24563 === 0) {
                    /** @type {number} */
                    local$$24560 = 0;
                    local$$24554 = local$$24554 + local$$24688;
                    local$$24667();
                    return;
                  } else {
                    local$$24563--;
                  }
                }
              }
              break;
            case ,:
              if (local$$24565) {
                break;
              } else {
                if (local$$24560 === 0) {
                  local$$24667();
                  return;
                } else {
                  if (local$$24560 === 1) {
                    if (local$$24563 === 0 && !local$$24546[match](/^url$/i)) {
                      local$$24567[push](local$$24548);
                      local$$24548 = ;
                      local$$24554 = local$$24554 + local$$24688;
                      return;
                    }
                  }
                }
              }
              break;
          }
          local$$24554 = local$$24554 + local$$24688;
          if (local$$24560 === 0) {
            local$$24546 = local$$24546 + local$$24688;
          } else {
            local$$24548 = local$$24548 + local$$24688;
          }
        });
        local$$24667();
        return local$$24557;
      };
    }, {}],
    27 : [function(local$$24854, local$$24855, local$$24856) {
      /**
       * @param {?} local$$24859
       * @return {undefined}
       */
      function local$$24858(local$$24859) {
        local$$24861[apply](this, arguments);
        this[type] = local$$24859[args][0] === linear ? local$$24861[TYPES][LINEAR] : local$$24861[TYPES][RADIAL];
      }
      var local$$24861 = local$$24854(./gradientcontainer);
      local$$24858[prototype] = Object[create](local$$24861[prototype]);
      /** @type {function(?): undefined} */
      local$$24855[exports] = local$$24858;
    }, {
      "./gradientcontainer" : 9
    }],
    28 : [function(local$$24922, local$$24923, local$$24924) {
      /**
       * @param {?} local$$24927
       * @return {?}
       */
      function local$$24926(local$$24927) {
        return new Promise(function(local$$24929, local$$24930) {
          /** @type {!XMLHttpRequest} */
          var local$$24933 = new XMLHttpRequest;
          local$$24933[open](GET, local$$24927);
          /**
           * @return {undefined}
           */
          local$$24933[onload] = function() {
            if (local$$24933[status] === 200) {
              local$$24929(local$$24933[responseText]);
            } else {
              local$$24930(new Error(local$$24933[statusText]));
            }
          };
          /**
           * @return {undefined}
           */
          local$$24933[onerror] = function() {
            local$$24930(new Error(Network Error));
          };
          local$$24933[send]();
        });
      }
      /** @type {function(?): ?} */
      local$$24923[exports] = local$$24926;
    }, {}]
  }, {}, [4])(4);
});
/**
 * @param {string} local$$25021
 * @return {undefined}
 */
THREE[MTLLoader] = function(local$$25021) {
  this[manager] = local$$25021 !== undefined ? local$$25021 : THREE[DefaultLoadingManager];
};
Object[assign](THREE[MTLLoader][prototype], THREE[EventDispatcher][prototype], {
  load : function(local$$25052, local$$25053, local$$25054, local$$25055) {
    var local$$25057 = this;
    var local$$25065 = new THREE.XHRLoader(this[manager]);
    local$$25065[setPath](this[path]);
    local$$25065[load](local$$25052, function(local$$25078) {
      local$$25053(local$$25057[parse](local$$25078));
    }, local$$25054, local$$25055);
  },
  setPath : function(local$$25092) {
    this[path] = local$$25092;
  },
  setTexturePath : function(local$$25101) {
    this[texturePath] = local$$25101;
  },
  setBaseUrl : function(local$$25110) {
    console[warn](THREE.MTLLoader: .setBaseUrl() is deprecated. Use .setTexturePath( path ) for texture path or .setPath( path ) for general base path instead.);
    this[setTexturePath](local$$25110);
  },
  setCrossOrigin : function(local$$25127) {
    this[crossOrigin] = local$$25127;
  },
  setMaterialOptions : function(local$$25136) {
    this[materialOptions] = local$$25136;
  },
  parse : function(local$$25146) {
    var local$$25154 = local$$25146[split](
);
    var local$$25157 = {};
    /** @type {!RegExp} */
    var local$$25160 = /\s+/;
    var local$$25163 = {};
    /** @type {number} */
    var local$$25166 = 0;
    for (; local$$25166 < local$$25154[length]; local$$25166++) {
      var local$$25175 = local$$25154[local$$25166];
      local$$25175 = local$$25175[trim]();
      if (local$$25175[length] === 0 || local$$25175[charAt](0) === #) {
        continue;
      }
      var local$$25207 = local$$25175[indexOf]( );
      var local$$25217 = local$$25207 >= 0 ? local$$25175[substring](0, local$$25207) : local$$25175;
      local$$25217 = local$$25217[toLowerCase]();
      var local$$25236 = local$$25207 >= 0 ? local$$25175[substring](local$$25207 + 1) : ;
      local$$25236 = local$$25236[trim]();
      if (local$$25217 === newmtl) {
        local$$25157 = {
          name : local$$25236
        };
        local$$25163[local$$25236] = local$$25157;
      } else {
        if (local$$25157) {
          if (local$$25217 === ka || local$$25217 === kd || local$$25217 === ks) {
            var local$$25270 = local$$25236[split](local$$25160, 3);
            /** @type {!Array} */
            local$$25157[local$$25217] = [parseFloat(local$$25270[0]), parseFloat(local$$25270[1]), parseFloat(local$$25270[2])];
          } else {
            local$$25157[local$$25217] = local$$25236;
          }
        }
      }
    }
    var local$$25313 = new THREE[MTLLoader].MaterialCreator(this[texturePath] || this[path], this[materialOptions]);
    local$$25313[setCrossOrigin](this[crossOrigin]);
    local$$25313[setManager](this[manager]);
    local$$25313[setMaterials](local$$25163);
    return local$$25313;
  }
});
/**
 * @param {?} local$$25348
 * @param {?} local$$25349
 * @return {undefined}
 */
THREE[MTLLoader][MaterialCreator] = function(local$$25348, local$$25349) {
  this[baseUrl] = local$$25348 || ;
  this[options] = local$$25349;
  this[materialsInfo] = {};
  this[materials] = {};
  /** @type {!Array} */
  this[materialsArray] = [];
  this[nameLookup] = {};
  this[side] = this[options] && this[options][side] ? this[options][side] : THREE[FrontSide];
  this[wrap] = this[options] && this[options][wrap] ? this[options][wrap] : THREE[RepeatWrapping];
};
THREE[MTLLoader][MaterialCreator][prototype] = {
  constructor : THREE[MTLLoader][MaterialCreator],
  setCrossOrigin : function(local$$25457) {
    this[crossOrigin] = local$$25457;
  },
  setManager : function(local$$25466) {
    this[manager] = local$$25466;
  },
  setMaterials : function(local$$25475) {
    this[materialsInfo] = this[convert](local$$25475);
    this[materials] = {};
    /** @type {!Array} */
    this[materialsArray] = [];
    this[nameLookup] = {};
  },
  convert : function(local$$25506) {
    if (!this[options]) {
      return local$$25506;
    }
    var local$$25517 = {};
    var local$$25519;
    for (local$$25519 in local$$25506) {
      var local$$25522 = local$$25506[local$$25519];
      var local$$25525 = {};
      local$$25517[local$$25519] = local$$25525;
      var local$$25530;
      for (local$$25530 in local$$25522) {
        /** @type {boolean} */
        var local$$25533 = true;
        var local$$25536 = local$$25522[local$$25530];
        var local$$25542 = local$$25530[toLowerCase]();
        switch(local$$25542) {
          case kd:
          case ka:
          case ks:
            if (this[options] && this[options][normalizeRGB]) {
              /** @type {!Array} */
              local$$25536 = [local$$25536[0] / 255, local$$25536[1] / 255, local$$25536[2] / 255];
            }
            if (this[options] && this[options][ignoreZeroRGBs]) {
              if (local$$25536[0] === 0 && local$$25536[1] === 0 && local$$25536[2] === 0) {
                /** @type {boolean} */
                local$$25533 = false;
              }
            }
            break;
          default:
            break;
        }
        if (local$$25533) {
          local$$25525[local$$25542] = local$$25536;
        }
      }
    }
    return local$$25517;
  },
  preload : function() {
    var local$$25638;
    for (local$$25638 in this[materialsInfo]) {
      this[create](local$$25638);
    }
  },
  getIndex : function(local$$25652) {
    return this[nameLookup][local$$25652];
  },
  getAsArray : function() {
    /** @type {number} */
    var local$$25664 = 0;
    var local$$25666;
    for (local$$25666 in this[materialsInfo]) {
      this[materialsArray][local$$25664] = this[create](local$$25666);
      /** @type {number} */
      this[nameLookup][local$$25666] = local$$25664;
      local$$25664++;
    }
    return this[materialsArray];
  },
  create : function(local$$25699) {
    if (this[materials][local$$25699] === undefined) {
      this[createMaterial_](local$$25699);
    }
    return this[materials][local$$25699];
  },
  createMaterial_ : function(local$$25721) {
    /**
     * @param {?} local$$25724
     * @param {?} local$$25725
     * @return {undefined}
     */
    function local$$25723(local$$25724, local$$25725) {
      if (local$$25727[local$$25724]) {
        return;
      }
      var local$$25738 = local$$25733[getTextureParams](local$$25725, local$$25727);
      var local$$25752 = local$$25733[loadTexture](local$$25743(local$$25733[baseUrl], local$$25738[url]));
      local$$25752[repeat][copy](local$$25738[scale]);
      local$$25752[offset][copy](local$$25738[offset]);
      local$$25752[wrapS] = local$$25733[wrap];
      local$$25752[wrapT] = local$$25733[wrap];
      local$$25727[local$$25724] = local$$25752;
    }
    var local$$25733 = this;
    var local$$25802 = this[materialsInfo][local$$25721];
    var local$$25727 = {
      name : local$$25721,
      side : this[side]
    };
    /**
     * @param {(Object|number)} local$$25809
     * @param {!Object} local$$25810
     * @return {?}
     */
    var local$$25743 = function(local$$25809, local$$25810) {
      if (typeof local$$25810 !== string || local$$25810 === ) {
        return ;
      }
      if (/^https?:\/\//i[test](local$$25810)) {
        return local$$25810;
      }
      return local$$25809 + local$$25810;
    };
    var local$$25841;
    for (local$$25841 in local$$25802) {
      var local$$25844 = local$$25802[local$$25841];
      if (local$$25844 === ) {
        continue;
      }
      switch(local$$25841[toLowerCase]()) {
        case kd:
          local$$25727[color] = (new THREE.Color)[fromArray](local$$25844);
          break;
        case ks:
          local$$25727[specular] = (new THREE.Color)[fromArray](local$$25844);
          break;
        case map_kd:
          local$$25723(map, local$$25844);
          break;
        case map_ks:
          local$$25723(specularMap, local$$25844);
          break;
        case map_bump:
        case bump:
          local$$25723(bumpMap, local$$25844);
          break;
        case ns:
          /** @type {number} */
          local$$25727[shininess] = parseFloat(local$$25844);
          break;
        case d:
          if (local$$25844 < 1) {
            local$$25727[opacity] = local$$25844;
            /** @type {boolean} */
            local$$25727[transparent] = true;
          }
          break;
        case Tr:
          if (local$$25844 > 0) {
            /** @type {number} */
            local$$25727[opacity] = 1 - local$$25844;
            /** @type {boolean} */
            local$$25727[transparent] = true;
          }
          break;
        default:
          break;
      }
    }
    this[materials][local$$25721] = new THREE.MeshPhongMaterial(local$$25727);
    return this[materials][local$$25721];
  },
  getTextureParams : function(local$$25999, local$$26000) {
    var local$$26011 = {
      scale : new THREE.Vector2(1, 1),
      offset : new THREE.Vector2(0, 0)
    };
    var local$$26018 = local$$25999[split](/\s+/);
    var local$$26020;
    local$$26020 = local$$26018[indexOf](-bm);
    if (local$$26020 >= 0) {
      /** @type {number} */
      local$$26000[bumpScale] = parseFloat(local$$26018[local$$26020 + 1]);
      local$$26018[splice](local$$26020, 2);
    }
    local$$26020 = local$$26018[indexOf](-s);
    if (local$$26020 >= 0) {
      local$$26011[scale][set](parseFloat(local$$26018[local$$26020 + 1]), parseFloat(local$$26018[local$$26020 + 2]));
      local$$26018[splice](local$$26020, 4);
    }
    local$$26020 = local$$26018[indexOf](-o);
    if (local$$26020 >= 0) {
      local$$26011[offset][set](parseFloat(local$$26018[local$$26020 + 1]), parseFloat(local$$26018[local$$26020 + 2]));
      local$$26018[splice](local$$26020, 4);
    }
    local$$26011[url] = local$$26018[join]( )[trim]();
    return local$$26011;
  },
  loadTexture : function(local$$26138, local$$26139, local$$26140, local$$26141, local$$26142) {
    var local$$26144;
    var local$$26156 = THREE[Loader][Handlers][get](local$$26138);
    var local$$26169 = this[manager] !== undefined ? this[manager] : THREE[DefaultLoadingManager];
    if (local$$26156 === null) {
      local$$26156 = new THREE.TextureLoader(local$$26169);
    }
    if (local$$26156[setCrossOrigin]) {
      local$$26156[setCrossOrigin](this[crossOrigin]);
    }
    local$$26144 = local$$26156[load](local$$26138, local$$26140, local$$26141, local$$26142);
    if (local$$26139 !== undefined) {
      /** @type {string} */
      local$$26144[mapping] = local$$26139;
    }
    return local$$26144;
  }
};
/**
 * @param {string} local$$26221
 * @return {undefined}
 */
THREE[OBJLoader] = function(local$$26221) {
  this[manager] = local$$26221 !== undefined ? local$$26221 : THREE[DefaultLoadingManager];
  /** @type {null} */
  this[materials] = null;
  this[regexp] = {
    vertex_pattern : /^v\s+([\d|\.|\+|\-|e|E]+)\s+([\d|\.|\+|\-|e|E]+)\s+([\d|\.|\+|\-|e|E]+)/,
    normal_pattern : /^vn\s+([\d|\.|\+|\-|e|E]+)\s+([\d|\.|\+|\-|e|E]+)\s+([\d|\.|\+|\-|e|E]+)/,
    uv_pattern : /^vt\s+([\d|\.|\+|\-|e|E]+)\s+([\d|\.|\+|\-|e|E]+)/,
    face_vertex : /^f\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)(?:\s+(-?\d+))?/,
    face_vertex_uv : /^f\s+(-?\d+)\/(-?\d+)\s+(-?\d+)\/(-?\d+)\s+(-?\d+)\/(-?\d+)(?:\s+(-?\d+)\/(-?\d+))?/,
    face_vertex_uv_normal : /^f\s+(-?\d+)\/(-?\d+)\/(-?\d+)\s+(-?\d+)\/(-?\d+)\/(-?\d+)\s+(-?\d+)\/(-?\d+)\/(-?\d+)(?:\s+(-?\d+)\/(-?\d+)\/(-?\d+))?/,
    face_vertex_normal : /^f\s+(-?\d+)\/\/(-?\d+)\s+(-?\d+)\/\/(-?\d+)\s+(-?\d+)\/\/(-?\d+)(?:\s+(-?\d+)\/\/(-?\d+))?/,
    object_pattern : /^[og]\s*(.+)?/,
    smoothing_pattern : /^s\s+(\d+|on|off)/,
    material_library_pattern : /^mtllib /,
    material_use_pattern : /^usemtl /
  };
};
THREE[OBJLoader][prototype] = {
  constructor : THREE[OBJLoader],
  load : function(local$$26280, local$$26281, local$$26282, local$$26283) {
    var local$$26285 = this;
    var local$$26292 = new THREE.XHRLoader(local$$26285[manager]);
    local$$26292[setPath](this[path]);
    local$$26292[load](local$$26280, function(local$$26305) {
      local$$26281(local$$26285[parse](local$$26305));
    }, local$$26282, local$$26283);
  },
  setPath : function(local$$26319) {
    this[path] = local$$26319;
  },
  setMaterials : function(local$$26328) {
    this[materials] = local$$26328;
  },
  _createParserState : function() {
    var local$$27475 = {
      objects : [],
      object : {},
      vertices : [],
      normals : [],
      uvs : [],
      materialLibraries : [],
      startObject : function(local$$26344, local$$26345) {
        if (this[object] && this[object][fromDeclaration] === false) {
          /** @type {string} */
          this[object][name] = local$$26344;
          /** @type {boolean} */
          this[object][fromDeclaration] = local$$26345 !== false;
          return;
        }
        var local$$26403 = this[object] && typeof this[object][currentMaterial] === function ? this[object][currentMaterial]() : undefined;
        if (this[object] && typeof this[object][_finalize] === function) {
          this[object]._finalize(true);
        }
        this[object] = {
          name : local$$26344 || ,
          fromDeclaration : local$$26345 !== false,
          geometry : {
            vertices : [],
            normals : [],
            uvs : []
          },
          materials : [],
          smooth : true,
          startMaterial : function(local$$26443, local$$26444) {
            var local$$26449 = this._finalize(false);
            if (local$$26449 && (local$$26449[inherited] || local$$26449[groupCount] <= 0)) {
              this[materials][splice](local$$26449[index], 1);
            }
            var local$$26563 = {
              index : this[materials][length],
              name : local$$26443 || ,
              mtllib : Array[isArray](local$$26444) && local$$26444[length] > 0 ? local$$26444[local$$26444[length] - 1] : ,
              smooth : local$$26449 !== undefined ? local$$26449[smooth] : this[smooth],
              groupStart : local$$26449 !== undefined ? local$$26449[groupEnd] : 0,
              groupEnd : -1,
              groupCount : -1,
              inherited : false,
              clone : function(local$$26521) {
                var local$$26545 = {
                  index : typeof local$$26521 === number ? local$$26521 : this[index],
                  name : this[name],
                  mtllib : this[mtllib],
                  smooth : this[smooth],
                  groupStart : 0,
                  groupEnd : -1,
                  groupCount : -1,
                  inherited : false
                };
                local$$26545[clone] = this[clone][bind](local$$26545);
                return local$$26545;
              }
            };
            this[materials][push](local$$26563);
            return local$$26563;
          },
          currentMaterial : function() {
            if (this[materials][length] > 0) {
              return this[materials][this[materials][length] - 1];
            }
            return undefined;
          },
          _finalize : function(local$$26604) {
            var local$$26610 = this[currentMaterial]();
            if (local$$26610 && local$$26610[groupEnd] === -1) {
              /** @type {number} */
              local$$26610[groupEnd] = this[geometry][vertices][length] / 3;
              /** @type {number} */
              local$$26610[groupCount] = local$$26610[groupEnd] - local$$26610[groupStart];
              /** @type {boolean} */
              local$$26610[inherited] = false;
            }
            if (local$$26604 && this[materials][length] > 1) {
              /** @type {number} */
              var local$$26672 = this[materials][length] - 1;
              for (; local$$26672 >= 0; local$$26672--) {
                if (this[materials][local$$26672][groupCount] <= 0) {
                  this[materials][splice](local$$26672, 1);
                }
              }
            }
            if (local$$26604 && this[materials][length] === 0) {
              this[materials][push]({
                name : ,
                smooth : this[smooth]
              });
            }
            return local$$26610;
          }
        };
        if (local$$26403 && local$$26403[name] && typeof local$$26403[clone] === function) {
          var local$$26752 = local$$26403[clone](0);
          /** @type {boolean} */
          local$$26752[inherited] = true;
          this[object][materials][push](local$$26752);
        }
        this[objects][push](this[object]);
      },
      finalize : function() {
        if (this[object] && typeof this[object][_finalize] === function) {
          this[object]._finalize(true);
        }
      },
      parseVertexIndex : function(local$$26813, local$$26814) {
        /** @type {number} */
        var local$$26818 = parseInt(local$$26813, 10);
        return (local$$26818 >= 0 ? local$$26818 - 1 : local$$26818 + local$$26814 / 3) * 3;
      },
      parseNormalIndex : function(local$$26833, local$$26834) {
        /** @type {number} */
        var local$$26838 = parseInt(local$$26833, 10);
        return (local$$26838 >= 0 ? local$$26838 - 1 : local$$26838 + local$$26834 / 3) * 3;
      },
      parseUVIndex : function(local$$26853, local$$26854) {
        /** @type {number} */
        var local$$26858 = parseInt(local$$26853, 10);
        return (local$$26858 >= 0 ? local$$26858 - 1 : local$$26858 + local$$26854 / 2) * 2;
      },
      addVertex : function(local$$26873, local$$26874, local$$26875) {
        var local$$26880 = this[vertices];
        var local$$26891 = this[object][geometry][vertices];
        local$$26891[push](local$$26880[local$$26873 + 0]);
        local$$26891[push](local$$26880[local$$26873 + 1]);
        local$$26891[push](local$$26880[local$$26873 + 2]);
        local$$26891[push](local$$26880[local$$26874 + 0]);
        local$$26891[push](local$$26880[local$$26874 + 1]);
        local$$26891[push](local$$26880[local$$26874 + 2]);
        local$$26891[push](local$$26880[local$$26875 + 0]);
        local$$26891[push](local$$26880[local$$26875 + 1]);
        local$$26891[push](local$$26880[local$$26875 + 2]);
      },
      addVertexLine : function(local$$26967) {
        var local$$26972 = this[vertices];
        var local$$26983 = this[object][geometry][vertices];
        local$$26983[push](local$$26972[local$$26967 + 0]);
        local$$26983[push](local$$26972[local$$26967 + 1]);
        local$$26983[push](local$$26972[local$$26967 + 2]);
      },
      addNormal : function(local$$27011, local$$27012, local$$27013) {
        var local$$27018 = this[normals];
        var local$$27029 = this[object][geometry][normals];
        local$$27029[push](local$$27018[local$$27011 + 0]);
        local$$27029[push](local$$27018[local$$27011 + 1]);
        local$$27029[push](local$$27018[local$$27011 + 2]);
        local$$27029[push](local$$27018[local$$27012 + 0]);
        local$$27029[push](local$$27018[local$$27012 + 1]);
        local$$27029[push](local$$27018[local$$27012 + 2]);
        local$$27029[push](local$$27018[local$$27013 + 0]);
        local$$27029[push](local$$27018[local$$27013 + 1]);
        local$$27029[push](local$$27018[local$$27013 + 2]);
      },
      addUV : function(local$$27105, local$$27106, local$$27107) {
        var local$$27112 = this[uvs];
        var local$$27123 = this[object][geometry][uvs];
        local$$27123[push](local$$27112[local$$27105 + 0]);
        local$$27123[push](local$$27112[local$$27105 + 1]);
        local$$27123[push](local$$27112[local$$27106 + 0]);
        local$$27123[push](local$$27112[local$$27106 + 1]);
        local$$27123[push](local$$27112[local$$27107 + 0]);
        local$$27123[push](local$$27112[local$$27107 + 1]);
      },
      addUVLine : function(local$$27175) {
        var local$$27180 = this[uvs];
        var local$$27191 = this[object][geometry][uvs];
        local$$27191[push](local$$27180[local$$27175 + 0]);
        local$$27191[push](local$$27180[local$$27175 + 1]);
      },
      addFace : function(local$$27211, local$$27212, local$$27213, local$$27214, local$$27215, local$$27216, local$$27217, local$$27218, local$$27219, local$$27220, local$$27221, local$$27222) {
        var local$$27230 = this[vertices][length];
        var local$$27236 = this[parseVertexIndex](local$$27211, local$$27230);
        var local$$27242 = this[parseVertexIndex](local$$27212, local$$27230);
        var local$$27248 = this[parseVertexIndex](local$$27213, local$$27230);
        var local$$27250;
        if (local$$27214 === undefined) {
          this[addVertex](local$$27236, local$$27242, local$$27248);
        } else {
          local$$27250 = this[parseVertexIndex](local$$27214, local$$27230);
          this[addVertex](local$$27236, local$$27242, local$$27250);
          this[addVertex](local$$27242, local$$27248, local$$27250);
        }
        if (local$$27215 !== undefined) {
          var local$$27285 = this[uvs][length];
          local$$27236 = this[parseUVIndex](local$$27215, local$$27285);
          local$$27242 = this[parseUVIndex](local$$27216, local$$27285);
          local$$27248 = this[parseUVIndex](local$$27217, local$$27285);
          if (local$$27214 === undefined) {
            this[addUV](local$$27236, local$$27242, local$$27248);
          } else {
            local$$27250 = this[parseUVIndex](local$$27218, local$$27285);
            this[addUV](local$$27236, local$$27242, local$$27250);
            this[addUV](local$$27242, local$$27248, local$$27250);
          }
        }
        if (local$$27219 !== undefined) {
          var local$$27340 = this[normals][length];
          local$$27236 = this[parseNormalIndex](local$$27219, local$$27340);
          local$$27242 = local$$27219 === local$$27220 ? local$$27236 : this[parseNormalIndex](local$$27220, local$$27340);
          local$$27248 = local$$27219 === local$$27221 ? local$$27236 : this[parseNormalIndex](local$$27221, local$$27340);
          if (local$$27214 === undefined) {
            this[addNormal](local$$27236, local$$27242, local$$27248);
          } else {
            local$$27250 = this[parseNormalIndex](local$$27222, local$$27340);
            this[addNormal](local$$27236, local$$27242, local$$27250);
            this[addNormal](local$$27242, local$$27248, local$$27250);
          }
        }
      },
      addLineGeometry : function(local$$27393, local$$27394) {
        this[object][geometry][type] = Line;
        var local$$27415 = this[vertices][length];
        var local$$27423 = this[uvs][length];
        /** @type {number} */
        var local$$27426 = 0;
        var local$$27431 = local$$27393[length];
        for (; local$$27426 < local$$27431; local$$27426++) {
          this[addVertexLine](this[parseVertexIndex](local$$27393[local$$27426], local$$27415));
        }
        /** @type {number} */
        var local$$27450 = 0;
        local$$27431 = local$$27394[length];
        for (; local$$27450 < local$$27431; local$$27450++) {
          this[addUVLine](this[parseUVIndex](local$$27394[local$$27450], local$$27423));
        }
      }
    };
    local$$27475[startObject](, false);
    return local$$27475;
  },
  parse : function(local$$27489) {
    console[time](OBJLoader);
    var local$$27500 = this._createParserState();
    if (local$$27489[indexOf](
) !== -1) {
      local$$27489 = local$$27489[replace](/\r\n/g, 
);
    }
    if (local$$27489[indexOf](\
) !== -1) {
      local$$27489 = local$$27489[replace](/\\\n/g, );
    }
    var local$$27550 = local$$27489[split](
);
    var local$$27554 = ;
    var local$$27558 = ;
    var local$$27562 = ;
    /** @type {number} */
    var local$$27565 = 0;
    /** @type {!Array} */
    var local$$27568 = [];
    /** @type {boolean} */
    var local$$27579 = typeof [trimLeft] === function;
    /** @type {number} */
    var local$$27582 = 0;
    var local$$27587 = local$$27550[length];
    for (; local$$27582 < local$$27587; local$$27582++) {
      local$$27554 = local$$27550[local$$27582];
      local$$27554 = local$$27579 ? local$$27554[trimLeft]() : local$$27554[trim]();
      local$$27565 = local$$27554[length];
      if (local$$27565 === 0) {
        continue;
      }
      local$$27558 = local$$27554[charAt](0);
      if (local$$27558 === #) {
        continue;
      }
      if (local$$27558 === v) {
        local$$27562 = local$$27554[charAt](1);
        if (local$$27562 ===   && (local$$27568 = this[regexp][vertex_pattern][exec](local$$27554)) !== null) {
          local$$27500[vertices][push](parseFloat(local$$27568[1]), parseFloat(local$$27568[2]), parseFloat(local$$27568[3]));
        } else {
          if (local$$27562 === n && (local$$27568 = this[regexp][normal_pattern][exec](local$$27554)) !== null) {
            local$$27500[normals][push](parseFloat(local$$27568[1]), parseFloat(local$$27568[2]), parseFloat(local$$27568[3]));
          } else {
            if (local$$27562 === t && (local$$27568 = this[regexp][uv_pattern][exec](local$$27554)) !== null) {
              local$$27500[uvs][push](parseFloat(local$$27568[1]), parseFloat(local$$27568[2]));
            } else {
              throw new Error(Unexpected vertex/normal/uv line: ' + local$$27554 + ');
            }
          }
        }
      } else {
        if (local$$27558 === f) {
          if ((local$$27568 = this[regexp][face_vertex_uv_normal][exec](local$$27554)) !== null) {
            local$$27500[addFace](local$$27568[1], local$$27568[4], local$$27568[7], local$$27568[10], local$$27568[2], local$$27568[5], local$$27568[8], local$$27568[11], local$$27568[3], local$$27568[6], local$$27568[9], local$$27568[12]);
          } else {
            if ((local$$27568 = this[regexp][face_vertex_uv][exec](local$$27554)) !== null) {
              local$$27500[addFace](local$$27568[1], local$$27568[3], local$$27568[5], local$$27568[7], local$$27568[2], local$$27568[4], local$$27568[6], local$$27568[8]);
            } else {
              if ((local$$27568 = this[regexp][face_vertex_normal][exec](local$$27554)) !== null) {
                local$$27500[addFace](local$$27568[1], local$$27568[3], local$$27568[5], local$$27568[7], undefined, undefined, undefined, undefined, local$$27568[2], local$$27568[4], local$$27568[6], local$$27568[8]);
              } else {
                if ((local$$27568 = this[regexp][face_vertex][exec](local$$27554)) !== null) {
                  local$$27500[addFace](local$$27568[1], local$$27568[2], local$$27568[3], local$$27568[4]);
                } else {
                  throw new Error(Unexpected face line: ' + local$$27554 + ');
                }
              }
            }
          }
        } else {
          if (local$$27558 === l) {
            var local$$27936 = local$$27554[substring](1)[trim]()[split]( );
            /** @type {!Array} */
            var local$$27939 = [];
            /** @type {!Array} */
            var local$$27942 = [];
            if (local$$27554[indexOf](/) === -1) {
              local$$27939 = local$$27936;
            } else {
              /** @type {number} */
              var local$$27956 = 0;
              var local$$27961 = local$$27936[length];
              for (; local$$27956 < local$$27961; local$$27956++) {
                var local$$27973 = local$$27936[local$$27956][split](/);
                if (local$$27973[0] !== ) {
                  local$$27939[push](local$$27973[0]);
                }
                if (local$$27973[1] !== ) {
                  local$$27942[push](local$$27973[1]);
                }
              }
            }
            local$$27500[addLineGeometry](local$$27939, local$$27942);
          } else {
            if ((local$$27568 = this[regexp][object_pattern][exec](local$$27554)) !== null) {
              var local$$28047 = (  + local$$27568[0][substr](1)[trim]())[substr](1);
              local$$27500[startObject](local$$28047);
            } else {
              if (this[regexp][material_use_pattern][test](local$$27554)) {
                local$$27500[object][startMaterial](local$$27554[substring](7)[trim](), local$$27500[materialLibraries]);
              } else {
                if (this[regexp][material_library_pattern][test](local$$27554)) {
                  local$$27500[materialLibraries][push](local$$27554[substring](7)[trim]());
                } else {
                  if ((local$$27568 = this[regexp][smoothing_pattern][exec](local$$27554)) !== null) {
                    var local$$28137 = local$$27568[1][trim]()[toLowerCase]();
                    /** @type {boolean} */
                    local$$27500[object][smooth] = local$$28137 === 1 || local$$28137 === on;
                    var local$$28161 = local$$27500[object][currentMaterial]();
                    if (local$$28161) {
                      local$$28161[smooth] = local$$27500[object][smooth];
                    }
                  } else {
                    if (local$$27554 ===  ) {
                      continue;
                    }
                    throw new Error(Unexpected line: ' + local$$27554 + ');
                  }
                }
              }
            }
          }
        }
      }
    }
    local$$27500[finalize]();
    var local$$28216 = new THREE.Group;
    local$$28216[materialLibraries] = [][concat](local$$27500[materialLibraries]);
    /** @type {number} */
    local$$27582 = 0;
    local$$27587 = local$$27500[objects][length];
    for (; local$$27582 < local$$27587; local$$27582++) {
      var local$$28249 = local$$27500[objects][local$$27582];
      var local$$28254 = local$$28249[geometry];
      var local$$28259 = local$$28249[materials];
      /** @type {boolean} */
      var local$$28267 = local$$28254[type] === Line;
      if (local$$28254[vertices][length] === 0) {
        continue;
      }
      var local$$28283 = new THREE.BufferGeometry;
      local$$28283[addAttribute](position, new THREE.BufferAttribute(new Float32Array(local$$28254[vertices]), 3));
      if (local$$28254[normals][length] > 0) {
        local$$28283[addAttribute](normal, new THREE.BufferAttribute(new Float32Array(local$$28254[normals]), 3));
      } else {
        local$$28283[computeVertexNormals]();
      }
      if (local$$28254[uvs][length] > 0) {
        local$$28283[addAttribute](uv, new THREE.BufferAttribute(new Float32Array(local$$28254[uvs]), 2));
      }
      /** @type {!Array} */
      var local$$28358 = [];
      /** @type {number} */
      var local$$28361 = 0;
      var local$$28366 = local$$28259[length];
      for (; local$$28361 < local$$28366; local$$28361++) {
        var local$$28372 = local$$28259[local$$28361];
        local$$28161 = undefined;
        if (this[materials] !== null) {
          local$$28161 = this[materials][create](local$$28372[name]);
          if (local$$28267 && local$$28161 && !(local$$28161 instanceof THREE[LineBasicMaterial])) {
            var local$$28402 = new THREE.LineBasicMaterial;
            local$$28402[copy](local$$28161);
            local$$28161 = local$$28402;
          }
        }
        if (!local$$28161) {
          local$$28161 = !local$$28267 ? new THREE.MeshPhongMaterial : new THREE.LineBasicMaterial;
          local$$28161[name] = local$$28372[name];
        }
        local$$28161[shading] = local$$28372[smooth] ? THREE[SmoothShading] : THREE[FlatShading];
        local$$28358[push](local$$28161);
      }
      var local$$28459;
      if (local$$28358[length] > 1) {
        /** @type {number} */
        local$$28361 = 0;
        local$$28366 = local$$28259[length];
        for (; local$$28361 < local$$28366; local$$28361++) {
          local$$28372 = local$$28259[local$$28361];
          local$$28283[addGroup](local$$28372[groupStart], local$$28372[groupCount], local$$28361);
        }
        var local$$28497 = new THREE.MultiMaterial(local$$28358);
        local$$28459 = !local$$28267 ? new THREE.Mesh(local$$28283, local$$28497) : new THREE.LineSegments(local$$28283, local$$28497);
      } else {
        local$$28459 = !local$$28267 ? new THREE.Mesh(local$$28283, local$$28358[0]) : new THREE.LineSegments(local$$28283, local$$28358[0]);
      }
      local$$28459[name] = local$$28249[name];
      local$$28216[add](local$$28459);
    }
    console[timeEnd](OBJLoader);
    return local$$28216;
  }
};
/**
 * @param {?} local$$28555
 * @param {?} local$$28556
 * @param {?} local$$28557
 * @param {?} local$$28558
 * @return {undefined}
 */
function LSJRectangle(local$$28555, local$$28556, local$$28557, local$$28558) {
  this[left] = local$$28555;
  this[top] = local$$28556;
  this[right] = local$$28557;
  this[bottom] = local$$28558;
}
/** @type {function(?, ?, ?, ?): undefined} */
LSJRectangle[prototype][constructor] = LSJRectangle;
/**
 * @param {?} local$$28596
 * @return {?}
 */
LSJRectangle[prototype][intersection] = function(local$$28596) {
  return this[right] >= local$$28596[left] && this[left] <= local$$28596[right] && this[top] >= local$$28596[bottom] && this[bottom] <= local$$28596[top];
};
/**
 * @return {undefined}
 */
THREE[DDSLoader] = function() {
  this[_parser] = THREE[DDSLoader][parse];
};
THREE[DDSLoader][prototype] = Object[create](THREE[CompressedTextureLoader][prototype]);
THREE[DDSLoader][prototype][constructor] = THREE[DDSLoader];
/**
 * @param {number} local$$28693
 * @param {boolean} local$$28694
 * @return {?}
 */
THREE[DDSLoader][parse] = function(local$$28693, local$$28694) {
  /**
   * @param {?} local$$28697
   * @return {?}
   */
  function local$$28696(local$$28697) {
    return local$$28697[charCodeAt](0) + (local$$28697[charCodeAt](1) << 8) + (local$$28697[charCodeAt](2) << 16) + (local$$28697[charCodeAt](3) << 24);
  }
  /**
   * @param {number} local$$28732
   * @return {?}
   */
  function local$$28731(local$$28732) {
    return String[fromCharCode](local$$28732 & 255, local$$28732 >> 8 & 255, local$$28732 >> 16 & 255, local$$28732 >> 24 & 255);
  }
  /**
   * @param {number} local$$28756
   * @param {number} local$$28757
   * @param {number} local$$28758
   * @param {number} local$$28759
   * @return {?}
   */
  function local$$28755(local$$28756, local$$28757, local$$28758, local$$28759) {
    /** @type {number} */
    var local$$28764 = local$$28758 * local$$28759 * 4;
    /** @type {!Uint8Array} */
    var local$$28768 = new Uint8Array(local$$28756, local$$28757, local$$28764);
    /** @type {!Uint8Array} */
    var local$$28771 = new Uint8Array(local$$28764);
    /** @type {number} */
    var local$$28774 = 0;
    /** @type {number} */
    var local$$28777 = 0;
    /** @type {number} */
    var local$$28780 = 0;
    for (; local$$28780 < local$$28759; local$$28780++) {
      /** @type {number} */
      var local$$28786 = 0;
      for (; local$$28786 < local$$28758; local$$28786++) {
        /** @type {number} */
        var local$$28792 = local$$28768[local$$28777];
        local$$28777++;
        /** @type {number} */
        var local$$28797 = local$$28768[local$$28777];
        local$$28777++;
        /** @type {number} */
        var local$$28802 = local$$28768[local$$28777];
        local$$28777++;
        /** @type {number} */
        var local$$28807 = local$$28768[local$$28777];
        local$$28777++;
        /** @type {number} */
        local$$28771[local$$28774] = local$$28802;
        local$$28774++;
        /** @type {number} */
        local$$28771[local$$28774] = local$$28797;
        local$$28774++;
        /** @type {number} */
        local$$28771[local$$28774] = local$$28792;
        local$$28774++;
        /** @type {number} */
        local$$28771[local$$28774] = local$$28807;
        local$$28774++;
      }
    }
    return local$$28771;
  }
  var local$$28845 = {
    mipmaps : [],
    width : 0,
    height : 0,
    format : null,
    mipmapCount : 1
  };
  /** @type {number} */
  var local$$28848 = 542327876;
  /** @type {number} */
  var local$$28851 = 1;
  /** @type {number} */
  var local$$28854 = 2;
  /** @type {number} */
  var local$$28857 = 4;
  /** @type {number} */
  var local$$28860 = 8;
  /** @type {number} */
  var local$$28863 = 4096;
  /** @type {number} */
  var local$$28866 = 131072;
  /** @type {number} */
  var local$$28869 = 524288;
  /** @type {number} */
  var local$$28872 = 8388608;
  /** @type {number} */
  var local$$28875 = 8;
  /** @type {number} */
  var local$$28878 = 4194304;
  /** @type {number} */
  var local$$28881 = 4096;
  /** @type {number} */
  var local$$28884 = 512;
  /** @type {number} */
  var local$$28887 = 1024;
  /** @type {number} */
  var local$$28890 = 2048;
  /** @type {number} */
  var local$$28893 = 4096;
  /** @type {number} */
  var local$$28896 = 8192;
  /** @type {number} */
  var local$$28899 = 16384;
  /** @type {number} */
  var local$$28902 = 32768;
  /** @type {number} */
  var local$$28905 = 2097152;
  /** @type {number} */
  var local$$28908 = 1;
  /** @type {number} */
  var local$$28911 = 2;
  /** @type {number} */
  var local$$28914 = 4;
  /** @type {number} */
  var local$$28917 = 64;
  /** @type {number} */
  var local$$28920 = 512;
  /** @type {number} */
  var local$$28923 = 131072;
  var local$$28928 = local$$28696(DXT1);
  var local$$28933 = local$$28696(DXT3);
  var local$$28938 = local$$28696(DXT5);
  var local$$28943 = local$$28696(ETC1);
  /** @type {number} */
  var local$$28946 = 31;
  /** @type {number} */
  var local$$28949 = 0;
  /** @type {number} */
  var local$$28952 = 1;
  /** @type {number} */
  var local$$28955 = 2;
  /** @type {number} */
  var local$$28958 = 3;
  /** @type {number} */
  var local$$28961 = 4;
  /** @type {number} */
  var local$$28964 = 7;
  /** @type {number} */
  var local$$28967 = 20;
  /** @type {number} */
  var local$$28970 = 21;
  /** @type {number} */
  var local$$28973 = 22;
  /** @type {number} */
  var local$$28976 = 23;
  /** @type {number} */
  var local$$28979 = 24;
  /** @type {number} */
  var local$$28982 = 25;
  /** @type {number} */
  var local$$28985 = 26;
  /** @type {number} */
  var local$$28988 = 27;
  /** @type {number} */
  var local$$28991 = 28;
  /** @type {number} */
  var local$$28994 = 29;
  /** @type {number} */
  var local$$28997 = 30;
  /** @type {!Int32Array} */
  var local$$29002 = new Int32Array(local$$28693, 0, local$$28946);
  if (local$$29002[local$$28949] !== local$$28848) {
    console[error](THREE.DDSLoader.parse: Invalid magic number in DDS header.);
    return local$$28845;
  }
  if (!local$$29002[local$$28967] & local$$28914) {
    console[error](THREE.DDSLoader.parse: Unsupported format, must contain a FourCC code.);
    return local$$28845;
  }
  var local$$29031;
  /** @type {number} */
  var local$$29034 = local$$29002[local$$28970];
  /** @type {boolean} */
  var local$$29037 = false;
  switch(local$$29034) {
    case local$$28928:
      /** @type {number} */
      local$$29031 = 8;
      local$$28845[format] = THREE[RGB_S3TC_DXT1_Format];
      break;
    case local$$28933:
      /** @type {number} */
      local$$29031 = 16;
      local$$28845[format] = THREE[RGBA_S3TC_DXT3_Format];
      break;
    case local$$28938:
      /** @type {number} */
      local$$29031 = 16;
      local$$28845[format] = THREE[RGBA_S3TC_DXT5_Format];
      break;
    case local$$28943:
      /** @type {number} */
      local$$29031 = 8;
      local$$28845[format] = THREE[RGB_ETC1_Format];
      break;
    default:
      if (local$$29002[local$$28973] === 32 && local$$29002[local$$28976] & 16711680 && local$$29002[local$$28979] & 65280 && local$$29002[local$$28982] & 255 && local$$29002[local$$28985] & 4278190080) {
        /** @type {boolean} */
        local$$29037 = true;
        /** @type {number} */
        local$$29031 = 64;
        local$$28845[format] = THREE[RGBAFormat];
      } else {
        console[error](THREE.DDSLoader.parse: Unsupported FourCC code , local$$28731(local$$29034));
        return local$$28845;
      }
  }
  /** @type {number} */
  local$$28845[mipmapCount] = 1;
  if (local$$29002[local$$28955] & local$$28866 && local$$28694 !== false) {
    local$$28845[mipmapCount] = Math[max](1, local$$29002[local$$28964]);
  }
  /** @type {number} */
  var local$$29170 = local$$29002[local$$28991];
  /** @type {boolean} */
  local$$28845[isCubemap] = local$$29170 & local$$28884 ? true : false;
  if (local$$28845[isCubemap] && (!(local$$29170 & local$$28887) || !(local$$29170 & local$$28890) || !(local$$29170 & local$$28893) || !(local$$29170 & local$$28896) || !(local$$29170 & local$$28899) || !(local$$29170 & local$$28902))) {
    console[error](THREE.DDSLoader.parse: Incomplete cubemap faces);
    return local$$28845;
  }
  /** @type {number} */
  local$$28845[width] = local$$29002[local$$28961];
  /** @type {number} */
  local$$28845[height] = local$$29002[local$$28958];
  /** @type {number} */
  var local$$29228 = local$$29002[local$$28952] + 4;
  /** @type {number} */
  var local$$29236 = local$$28845[isCubemap] ? 6 : 1;
  /** @type {number} */
  var local$$29239 = 0;
  for (; local$$29239 < local$$29236; local$$29239++) {
    var local$$29247 = local$$28845[width];
    var local$$29252 = local$$28845[height];
    /** @type {number} */
    var local$$29255 = 0;
    for (; local$$29255 < local$$28845[mipmapCount]; local$$29255++) {
      if (local$$29037) {
        var local$$29264 = local$$28755(local$$28693, local$$29228, local$$29247, local$$29252);
        var local$$29269 = local$$29264[length];
      } else {
        /** @type {number} */
        local$$29269 = Math[max](4, local$$29247) / 4 * Math[max](4, local$$29252) / 4 * local$$29031;
        /** @type {!Uint8Array} */
        local$$29264 = new Uint8Array(local$$28693, local$$29228, local$$29269);
      }
      var local$$29297 = {
        "data" : local$$29264,
        "width" : local$$29247,
        "height" : local$$29252
      };
      local$$28845[mipmaps][push](local$$29297);
      local$$29228 = local$$29228 + local$$29269;
      local$$29247 = Math[max](local$$29247 >> 1, 1);
      local$$29252 = Math[max](local$$29252 >> 1, 1);
    }
  }
  return local$$28845;
};
/**
 * @param {string} local$$29343
 * @return {undefined}
 */
THREE[PVRLoader] = function(local$$29343) {
  this[manager] = local$$29343 !== undefined ? local$$29343 : THREE[DefaultLoadingManager];
  this[_parser] = THREE[PVRLoader][parse];
};
THREE[PVRLoader][prototype] = Object[create](THREE[CompressedTextureLoader][prototype]);
THREE[PVRLoader][prototype][constructor] = THREE[PVRLoader];
/**
 * @param {number} local$$29408
 * @param {?} local$$29409
 * @return {?}
 */
THREE[PVRLoader][parse] = function(local$$29408, local$$29409) {
  /** @type {number} */
  var local$$29412 = 13;
  /** @type {!Uint32Array} */
  var local$$29417 = new Uint32Array(local$$29408, 0, local$$29412);
  var local$$29421 = {
    buffer : local$$29408,
    header : local$$29417,
    loadMipmaps : local$$29409
  };
  if (local$$29417[0] === 55727696) {
    return THREE[PVRLoader]._parseV3(local$$29421);
  } else {
    if (local$$29417[11] === 559044176) {
      return THREE[PVRLoader]._parseV2(local$$29421);
    } else {
      throw new Error([THREE.PVRLoader] Unknown PVR format);
    }
  }
};
/**
 * @param {?} local$$29463
 * @return {?}
 */
THREE[PVRLoader][_parseV3] = function(local$$29463) {
  var local$$29468 = local$$29463[header];
  var local$$29470;
  var local$$29472;
  var local$$29476 = local$$29468[12];
  var local$$29480 = local$$29468[2];
  var local$$29484 = local$$29468[6];
  var local$$29488 = local$$29468[7];
  var local$$29492 = local$$29468[9];
  var local$$29496 = local$$29468[10];
  var local$$29500 = local$$29468[11];
  switch(local$$29480) {
    case 0:
      /** @type {number} */
      local$$29470 = 2;
      local$$29472 = THREE[RGB_PVRTC_2BPPV1_Format];
      break;
    case 1:
      /** @type {number} */
      local$$29470 = 2;
      local$$29472 = THREE[RGBA_PVRTC_2BPPV1_Format];
      break;
    case 2:
      /** @type {number} */
      local$$29470 = 4;
      local$$29472 = THREE[RGB_PVRTC_4BPPV1_Format];
      break;
    case 3:
      /** @type {number} */
      local$$29470 = 4;
      local$$29472 = THREE[RGBA_PVRTC_4BPPV1_Format];
      break;
    default:
      throw new Error(pvrtc - unsupported PVR format  + local$$29480);
  }
  local$$29463[dataPtr] = 52 + local$$29476;
  /** @type {number} */
  local$$29463[bpp] = local$$29470;
  local$$29463[format] = local$$29472;
  local$$29463[width] = local$$29488;
  local$$29463[height] = local$$29484;
  local$$29463[numSurfaces] = local$$29496;
  local$$29463[numMipmaps] = local$$29500;
  /** @type {boolean} */
  local$$29463[isCubemap] = local$$29496 === 6;
  return THREE[PVRLoader]._extract(local$$29463);
};
/**
 * @param {?} local$$29619
 * @return {?}
 */
THREE[PVRLoader][_parseV2] = function(local$$29619) {
  var local$$29624 = local$$29619[header];
  var local$$29628 = local$$29624[0];
  var local$$29632 = local$$29624[1];
  var local$$29636 = local$$29624[2];
  var local$$29640 = local$$29624[3];
  var local$$29644 = local$$29624[4];
  var local$$29648 = local$$29624[5];
  var local$$29652 = local$$29624[6];
  var local$$29656 = local$$29624[7];
  var local$$29660 = local$$29624[8];
  var local$$29664 = local$$29624[9];
  var local$$29668 = local$$29624[10];
  var local$$29672 = local$$29624[11];
  var local$$29676 = local$$29624[12];
  /** @type {number} */
  var local$$29679 = 255;
  /** @type {number} */
  var local$$29682 = 24;
  /** @type {number} */
  var local$$29685 = 25;
  /** @type {number} */
  var local$$29688 = local$$29644 & local$$29679;
  var local$$29690;
  /** @type {boolean} */
  var local$$29694 = local$$29668 > 0;
  if (local$$29688 === local$$29685) {
    local$$29690 = local$$29694 ? THREE[RGBA_PVRTC_4BPPV1_Format] : THREE[RGB_PVRTC_4BPPV1_Format];
    /** @type {number} */
    local$$29652 = 4;
  } else {
    if (local$$29688 === local$$29682) {
      local$$29690 = local$$29694 ? THREE[RGBA_PVRTC_2BPPV1_Format] : THREE[RGB_PVRTC_2BPPV1_Format];
      /** @type {number} */
      local$$29652 = 2;
    } else {
      throw new Error(pvrtc - unknown format  + local$$29688);
    }
  }
  local$$29619[dataPtr] = local$$29628;
  /** @type {number} */
  local$$29619[bpp] = local$$29652;
  local$$29619[format] = local$$29690;
  local$$29619[width] = local$$29636;
  local$$29619[height] = local$$29632;
  local$$29619[numSurfaces] = local$$29676;
  local$$29619[numMipmaps] = local$$29640 + 1;
  /** @type {boolean} */
  local$$29619[isCubemap] = local$$29676 === 6;
  return THREE[PVRLoader]._extract(local$$29619);
};
/**
 * @param {?} local$$29794
 * @return {?}
 */
THREE[PVRLoader][_extract] = function(local$$29794) {
  var local$$29813 = {
    mipmaps : [],
    width : local$$29794[width],
    height : local$$29794[height],
    format : local$$29794[format],
    mipmapCount : local$$29794[numMipmaps],
    isCubemap : local$$29794[isCubemap]
  };
  var local$$29818 = local$$29794[buffer];
  var local$$29823 = local$$29794[dataPtr];
  var local$$29828 = local$$29794[bpp];
  var local$$29833 = local$$29794[numSurfaces];
  /** @type {number} */
  var local$$29836 = 0;
  /** @type {number} */
  var local$$29839 = 0;
  /** @type {number} */
  var local$$29842 = 0;
  /** @type {number} */
  var local$$29845 = 0;
  /** @type {number} */
  var local$$29848 = 0;
  /** @type {number} */
  var local$$29851 = 0;
  if (local$$29828 === 2) {
    /** @type {number} */
    local$$29842 = 8;
    /** @type {number} */
    local$$29845 = 4;
  } else {
    /** @type {number} */
    local$$29842 = 4;
    /** @type {number} */
    local$$29845 = 4;
  }
  /** @type {number} */
  local$$29839 = local$$29842 * local$$29845 * local$$29828 / 8;
  /** @type {number} */
  local$$29813[mipmaps][length] = local$$29794[numMipmaps] * local$$29833;
  /** @type {number} */
  var local$$29890 = 0;
  for (; local$$29890 < local$$29794[numMipmaps];) {
    /** @type {number} */
    var local$$29902 = local$$29794[width] >> local$$29890;
    /** @type {number} */
    var local$$29908 = local$$29794[height] >> local$$29890;
    /** @type {number} */
    local$$29848 = local$$29902 / local$$29842;
    /** @type {number} */
    local$$29851 = local$$29908 / local$$29845;
    if (local$$29848 < 2) {
      /** @type {number} */
      local$$29848 = 2;
    }
    if (local$$29851 < 2) {
      /** @type {number} */
      local$$29851 = 2;
    }
    /** @type {number} */
    local$$29836 = local$$29848 * local$$29851 * local$$29839;
    /** @type {number} */
    var local$$29937 = 0;
    for (; local$$29937 < local$$29833; local$$29937++) {
      /** @type {!Uint8Array} */
      var local$$29943 = new Uint8Array(local$$29818, local$$29823, local$$29836);
      var local$$29946 = {
        data : local$$29943,
        width : local$$29902,
        height : local$$29908
      };
      local$$29813[mipmaps][local$$29937 * local$$29794[numMipmaps] + local$$29890] = local$$29946;
      local$$29823 = local$$29823 + local$$29836;
    }
    local$$29890++;
  }
  return local$$29813;
};
/**
 * @param {?} local$$29978
 * @return {?}
 */
function checkLicense(local$$29978) {
  var local$$29986 = window[navigator][appName];
  var local$$29994 = window[navigator][appVersion];
  var local$$30002 = window[navigator][language];
  var local$$30010 = window[navigator][appCodeName];
  var local$$30018 = window[navigator][platform];
  var local$$30029 = window[document][location][href];
  var local$$30051 = local$$29986 + # + local$$29994 + # + local$$30002 + # + local$$30010 + # + local$$30018 + # + local$$30029;
  var local$$30063 = guid= + local$$29978 + &url= + local$$30029 + &type=1;
  /** @type {boolean} */
  this[result] = false;
  var local$$30071 = this;
  try {
    /** @type {!XMLHttpRequest} */
    var local$$30074 = new XMLHttpRequest;
    local$$30074[open](POST, http://www.wish3d.com/license/getLicenseState.action, false);
    local$$30074[setRequestHeader](Content-type, application/x-www-form-urlencoded);
    /**
     * @return {?}
     */
    local$$30074[onreadystatechange] = function() {
      /** @type {!XMLHttpRequest} */
      var local$$30099 = local$$30074;
      if (local$$30099[readyState] == 4) {
        if (local$$30099[status] == 200) {
          var local$$30114 = local$$30099[responseText];
          if (local$$30114[trim]() == 1) {
            /** @type {boolean} */
            local$$30071[result] = true;
            return true;
          }
        } else {
          /** @type {boolean} */
          local$$30071[result] = false;
          return false;
        }
      }
    };
    local$$30074[send](local$$30063);
  } catch (local$$30156) {
  }
  return this[result];
}
/**
 * @param {?} local$$30169
 * @return {?}
 */
function checkPrivateLicense(local$$30169) {
  /** @type {boolean} */
  this[result] = false;
  var local$$30177 = this;
  try {
    /** @type {!XMLHttpRequest} */
    var local$$30180 = new XMLHttpRequest;
    local$$30180[open](POST, ../license/getHardwareComputerID.action, false);
    local$$30180[setRequestHeader](Content-type, application/x-www-form-urlencoded);
    /**
     * @return {?}
     */
    local$$30180[onreadystatechange] = function() {
      /** @type {!XMLHttpRequest} */
      var local$$30205 = local$$30180;
      if (local$$30205[readyState] == 4) {
        if (local$$30205[status] == 200) {
          var local$$30220 = local$$30205[responseText];
          if (local$$30220[trim]() == local$$30169) {
            /** @type {boolean} */
            local$$30177[result] = true;
            return true;
          } else {
            /** @type {boolean} */
            local$$30177[result] = false;
            return false;
          }
        } else {
          /** @type {boolean} */
          local$$30177[result] = false;
          return false;
        }
      }
    };
    local$$30180[send]();
  } catch (local$$30269) {
  }
  return this[result];
}
/**
 * @return {undefined}
 */
LSJCamera = function() {
  this[quaternion] = new THREE.Quaternion;
  this[position] = new THREE.Vector3;
};
/** @type {function(): undefined} */
LSJCamera[prototype][constructor] = LSJCamera;
/**
 * @return {?}
 */
LSJCamera[prototype][getPosition] = function() {
  return this[position];
};
/**
 * @param {?} local$$30331
 * @return {undefined}
 */
LSJCamera[prototype][setPosition] = function(local$$30331) {
  this[position][copy](local$$30331);
};
/**
 * @return {?}
 */
LSJCamera[prototype][getQuaternion] = function() {
  return this[quaternion];
};
/**
 * @param {?} local$$30366
 * @return {undefined}
 */
LSJCamera[prototype][setQuaternion] = function(local$$30366) {
  this[quaternion][copy](local$$30366);
};
/**
 * @param {?} local$$30383
 * @return {undefined}
 */
LSJCameraTrackControls = function(local$$30383) {
  this[camera] = local$$30383;
  /** @type {boolean} */
  this[enable] = false;
  /** @type {!Array} */
  this[cameraKeys] = [];
  /** @type {number} */
  this[looptime] = 0;
  var local$$30408 = this;
  this[vectorKeyframeTrack] = undefined;
  this[quaternionKeyframeTrack] = undefined;
  this[interpolant] = undefined;
  this[quaInterpolant] = undefined;
  /** @type {boolean} */
  this[bPause] = false;
  this[startTime] = Date[now]();
  this[pauseTime] = Date[now]();
  /**
   * @return {undefined}
   */
  this[init] = function() {
    /** @type {!Array} */
    var local$$30459 = [];
    /** @type {!Array} */
    var local$$30462 = [];
    /** @type {!Array} */
    var local$$30465 = [];
    /** @type {number} */
    var local$$30468 = 0;
    for (; local$$30468 < this[cameraKeys][length]; local$$30468++) {
      var local$$30483 = this[cameraKeys][local$$30468];
      local$$30459[push](local$$30483[time]);
      local$$30483[value][getPosition]()[toArray](local$$30462, local$$30462[length]);
      local$$30483[value][getQuaternion]()[toArray](local$$30465, local$$30465[length]);
      this[looptime] = local$$30483[time];
    }
    this[vectorKeyframeTrack] = new THREE.VectorKeyframeTrack(flytoCamera, local$$30459, local$$30462);
    this[interpolant] = this[vectorKeyframeTrack][createInterpolant](undefined);
    this[quaternionKeyframeTrack] = new THREE.QuaternionKeyframeTrack(flyToCamera, local$$30459, local$$30465);
    this[quaInterpolant] = this[quaternionKeyframeTrack][createInterpolant](undefined);
  };
  /**
   * @return {?}
   */
  this[getAllKeys] = function() {
    return this[cameraKeys];
  };
  /**
   * @return {?}
   */
  this[clear] = function() {
    return this[cameraKeys] = [];
  };
  /**
   * @return {undefined}
   */
  this[play] = function() {
    if (this[enable]) {
      return;
    }
    if (!this[bPause]) {
      this[init]();
      /** @type {boolean} */
      this[enable] = true;
      this[startTime] = Date[now]();
    } else {
      /** @type {boolean} */
      this[enable] = true;
      /** @type {number} */
      this[startTime] = Date[now]() - (this[pauseTime] - this[startTime]);
      /** @type {boolean} */
      this[bPause] = false;
    }
  };
  /**
   * @return {undefined}
   */
  this[stop] = function() {
    /** @type {boolean} */
    this[enable] = false;
    /** @type {number} */
    this[startTime] = 0;
    /** @type {boolean} */
    this[bPause] = false;
  };
  /**
   * @return {undefined}
   */
  this[pause] = function() {
    if (!this[bPause]) {
      /** @type {boolean} */
      this[enable] = false;
      /** @type {boolean} */
      this[bPause] = true;
      this[pauseTime] = Date[now]();
    }
  };
  /**
   * @return {undefined}
   */
  this[update] = function() {
    if (!this[enable]) {
      return;
    }
    var local$$30756 = Date[now]();
    if (local$$30756 - this[startTime] > this[looptime]) {
      /** @type {boolean} */
      this[enable] = false;
      var local$$30784 = this[cameraKeys][this[cameraKeys][length] - 1];
      local$$30408[camera][position][copy](local$$30784[position]);
      local$$30408[camera][quaternion][copy](local$$30784[quaternion]);
      return;
    }
    var local$$30829 = this[interpolant][evaluate](local$$30756 - this[startTime]);
    local$$30408[camera][position][fromArray](local$$30829);
    var local$$30853 = this[quaInterpolant][evaluate](local$$30756 - this[startTime]);
    local$$30408[camera][quaternion][fromArray](local$$30853);
    local$$30408[camera][updateMatrixWorld]();
  };
};
/**
 * @param {?} local$$30885
 * @param {!Array} local$$30886
 * @return {undefined}
 */
LSJBillboardPlugin = function(local$$30885, local$$30886) {
  /**
   * @return {undefined}
   */
  function local$$30888() {
    /** @type {!Float32Array} */
    var local$$30908 = new Float32Array([0, -5, 0, 0, 40.5, -5, 1, 0, 40.5, 5, 1, 1, 0, 5, 0, 1]);
    /** @type {!Uint16Array} */
    var local$$30919 = new Uint16Array([0, 1, 2, 0, 2, 3]);
    local$$30921 = local$$30922[createBuffer]();
    local$$30929 = local$$30922[createBuffer]();
    local$$30922[bindBuffer](local$$30922.ARRAY_BUFFER, local$$30921);
    local$$30922[bufferData](local$$30922.ARRAY_BUFFER, local$$30908, local$$30922.STATIC_DRAW);
    local$$30922[bindBuffer](local$$30922.ELEMENT_ARRAY_BUFFER, local$$30929);
    local$$30922[bufferData](local$$30922.ELEMENT_ARRAY_BUFFER, local$$30919, local$$30922.STATIC_DRAW);
    local$$30965 = local$$30966();
    local$$30970 = {
      position : local$$30922[getAttribLocation](local$$30965, position),
      uv : local$$30922[getAttribLocation](local$$30965, uv)
    };
    local$$30986 = {
      uvOffset : local$$30922[getUniformLocation](local$$30965, uvOffset),
      uvScale : local$$30922[getUniformLocation](local$$30965, uvScale),
      rotation : local$$30922[getUniformLocation](local$$30965, rotation),
      scale : local$$30922[getUniformLocation](local$$30965, scale),
      color : local$$30922[getUniformLocation](local$$30965, color),
      map : local$$30922[getUniformLocation](local$$30965, map),
      opacity : local$$30922[getUniformLocation](local$$30965, opacity),
      modelViewMatrix : local$$30922[getUniformLocation](local$$30965, modelViewMatrix),
      projectionMatrix : local$$30922[getUniformLocation](local$$30965, projectionMatrix),
      alphaTest : local$$30922[getUniformLocation](local$$30965, alphaTest)
    };
    var local$$31056 = document[createElement](canvas);
    /** @type {number} */
    local$$31056[width] = 8;
    /** @type {number} */
    local$$31056[height] = 8;
    var local$$31076 = local$$31056[getContext](2d);
    local$$31076[fillStyle] = white;
    local$$31076[fillRect](0, 0, 8, 8);
    local$$31094 = new THREE.Texture(local$$31056);
    /** @type {boolean} */
    local$$31094[needsUpdate] = true;
  }
  /**
   * @return {?}
   */
  function local$$30966() {
    var local$$31113 = local$$30922[createProgram]();
    var local$$31121 = local$$30922[createShader](local$$30922.VERTEX_SHADER);
    var local$$31129 = local$$30922[createShader](local$$30922.FRAGMENT_SHADER);
    local$$30922[shaderSource](local$$31121, [precision  + local$$30885[getPrecision]() +  float;, uniform mat4 modelViewMatrix;, uniform mat4 projectionMatrix;, uniform float rotation;, uniform vec2 scale;, uniform vec2 uvOffset;, uniform vec2 uvScale;, attribute vec2 position;, attribute vec2 uv;, varying vec2 vUV;, void main() {, vUV = uvOffset + uv * uvScale;, vec2 alignedPosition = position * scale;, vec2 rotatedPosition;, rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;, rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;, vec4 finalPosition;, finalPosition = modelViewMatrix * vec4( 0.0, 0.0, 0.0, 1.0 );, finalPosition.xy += rotatedPosition;, finalPosition = projectionMatrix * finalPosition;, gl_Position = finalPosition;, }][join](
));
    local$$30922[shaderSource](local$$31129, [precision  + local$$30885[getPrecision]() +  float;, uniform vec3 color;, uniform sampler2D map;, uniform float opacity;, uniform float alphaTest;, varying vec2 vUV;, void main() {, vec4 texture = texture2D( map, vUV );, if ( texture.a < alphaTest ) discard;, gl_FragColor = vec4( color * texture.xyz, texture.a * opacity );, }][join](
));
    local$$30922[compileShader](local$$31121);
    local$$30922[compileShader](local$$31129);
    local$$30922[attachShader](local$$31113, local$$31121);
    local$$30922[attachShader](local$$31113, local$$31129);
    local$$30922[linkProgram](local$$31113);
    return local$$31113;
  }
  /**
   * @param {?} local$$31266
   * @param {?} local$$31267
   * @return {?}
   */
  function local$$31265(local$$31266, local$$31267) {
    if (local$$31266[z] !== local$$31267[z]) {
      return local$$31267[z] - local$$31266[z];
    } else {
      return local$$31267[id] - local$$31266[id];
    }
  }
  /**
   * @param {!Array} local$$31298
   * @param {?} local$$31299
   * @return {?}
   */
  function local$$31297(local$$31298, local$$31299) {
    /** @type {number} */
    var local$$31302 = 0;
    var local$$31307 = local$$31298[length];
    for (; local$$31302 < local$$31307; local$$31302++) {
      var local$$31313 = local$$31298[local$$31302];
      var local$$31319 = local$$31313[intersection](local$$31299);
      if (local$$31319) {
        return true;
      }
    }
    return false;
  }
  var local$$30922 = local$$30885[context];
  var local$$31339 = local$$30885[state];
  this[objects] = local$$30885[objects];
  local$$30886 = local$$30886;
  var local$$30921;
  var local$$30929;
  var local$$30965;
  var local$$30970;
  var local$$30986;
  var local$$31094;
  var local$$31359 = new THREE.Vector3;
  var local$$31363 = new THREE.Quaternion;
  var local$$31367 = new THREE.Vector3;
  /**
   * @param {?} local$$31372
   * @param {?} local$$31373
   * @param {?} local$$31374
   * @return {undefined}
   */
  this[render] = function(local$$31372, local$$31373, local$$31374) {
    local$$30886 = local$$31374[billboards];
    if (local$$30886[length] === 0) {
      return;
    }
    if (local$$30965 === undefined) {
      local$$30888();
    }
    local$$30922[useProgram](local$$30965);
    local$$31339[initAttributes]();
    local$$31339[disableUnusedAttributes]();
    local$$31339[disable](local$$30922.CULL_FACE);
    local$$31339[enable](local$$30922.BLEND);
    local$$30922[bindBuffer](local$$30922.ELEMENT_ARRAY_BUFFER, local$$30929);
    local$$30922[uniformMatrix4fv](local$$30986[projectionMatrix], false, local$$31373[projectionMatrix][elements]);
    local$$31339[activeTexture](local$$30922.TEXTURE0);
    local$$30922[uniform1i](local$$30986[map], 0);
    local$$30922[uniform1i](local$$30986[fogType], 0);
    /** @type {number} */
    var local$$31472 = 0;
    var local$$31477 = local$$30886[length];
    for (; local$$31472 < local$$31477; local$$31472++) {
      var local$$31483 = local$$30886[local$$31472];
      this[objects][update](local$$31483);
      local$$31483[modelViewMatrix][multiplyMatrices](local$$31373[matrixWorldInverse], local$$31483[matrixWorld]);
      /** @type {number} */
      local$$31483[z] = -local$$31483[modelViewMatrix][elements][14];
    }
    local$$30886[sort](local$$31265);
    /** @type {!Array} */
    var local$$31530 = [];
    /** @type {!Array} */
    var local$$31533 = [];
    /** @type {number} */
    local$$31472 = local$$30886[length] - 1;
    for (; local$$31472 >= 0; local$$31472--) {
      local$$31483 = local$$30886[local$$31472];
      var local$$31553 = local$$31483[getScreenRect]();
      if (local$$31483[geoParent][type] != GeoLabel) {
        if (local$$31297(local$$31533, local$$31553)) {
          if (local$$31483[geoParent][style][iconPath] == undefined || local$$31483[geoParent][style][iconPath] == ) {
            continue;
          }
          local$$31483[geoParent][setNameVisble](false);
        } else {
          local$$31483[geoParent][setNameVisble](true);
        }
        local$$31533[push](local$$31553);
      }
      var local$$31628 = local$$31483[geometry][attributes];
      var local$$31630;
      for (local$$31630 in local$$30970) {
        var local$$31633 = local$$30970[local$$31630];
        if (local$$31633 >= 0) {
          var local$$31638 = local$$31628[local$$31630];
          if (local$$31638 !== undefined) {
            var local$$31644 = local$$31638[itemSize];
            var local$$31653 = this[objects][getAttributeBuffer](local$$31638);
            local$$31339[enableAttribute](local$$31633);
            local$$30922[bindBuffer](local$$30922.ARRAY_BUFFER, local$$31653);
            local$$30922[vertexAttribPointer](local$$31633, local$$31644, local$$30922.FLOAT, false, 0, 0);
          }
        }
      }
      var local$$31686 = local$$31483[material];
      local$$30922[uniform1f](local$$30986[alphaTest], local$$31686[alphaTest]);
      local$$30922[uniformMatrix4fv](local$$30986[modelViewMatrix], false, local$$31483[modelViewMatrix][elements]);
      local$$31483[matrixWorld][decompose](local$$31359, local$$31363, local$$31367);
      local$$31530[0] = local$$31367[x];
      local$$31530[1] = local$$31367[y];
      if (local$$31686[map] !== null) {
        local$$30922[uniform2f](local$$30986[uvOffset], local$$31686[map][offset][x], local$$31686[map][offset][y]);
        local$$30922[uniform2f](local$$30986[uvScale], local$$31686[map][repeat][x], local$$31686[map][repeat][y]);
      } else {
        local$$30922[uniform2f](local$$30986[uvOffset], 0, 0);
        local$$30922[uniform2f](local$$30986[uvScale], 1, 1);
      }
      local$$30922[uniform1f](local$$30986[opacity], local$$31686[opacity]);
      local$$30922[uniform3f](local$$30986[color], local$$31686[color][r], local$$31686[color][g], local$$31686[color][b]);
      local$$30922[uniform1f](local$$30986[rotation], local$$31686[rotation]);
      local$$30922[uniform2fv](local$$30986[scale], local$$31530);
      local$$31339[setBlending](local$$31686[blending], local$$31686[blendEquation], local$$31686[blendSrc], local$$31686[blendDst]);
      local$$31339[setDepthTest](local$$31686[depthTest]);
      local$$31339[setDepthWrite](local$$31686[depthWrite]);
      if (local$$31686[map] && local$$31686[map][image] && local$$31686[map][image][width]) {
        local$$30885[setTexture](local$$31686[map], 0);
      } else {
        local$$30885[setTexture](local$$31094, 0);
      }
      local$$30922[drawElements](local$$30922.TRIANGLES, 6, local$$30922.UNSIGNED_SHORT, 0);
    }
    local$$31339[enable](local$$30922.CULL_FACE);
    local$$30885[resetGLState]();
  };
};
/**
 * @param {string} local$$31981
 * @param {?} local$$31982
 * @param {string} local$$31983
 * @return {undefined}
 */
LSJBillboard = function(local$$31981, local$$31982, local$$31983) {
  THREE[Object3D][call](this);
  /** @type {!Uint16Array} */
  var local$$32001 = new Uint16Array([0, 1, 2, 0, 2, 3]);
  /** @type {number} */
  var local$$32004 = 1;
  /** @type {number} */
  var local$$32026 = local$$32004 * local$$31981[map][image][width] / local$$31981[map][image][height];
  if (local$$31983 != undefined) {
    var local$$32038 = local$$31983[verticalOrign] != undefined ? local$$31983[verticalOrign] : 0;
    /** @type {number} */
    var local$$32050 = local$$31983[horizontalOrigin] != undefined ? local$$31983[horizontalOrigin] * local$$32026 : 0;
    /** @type {!Float32Array} */
    var local$$32066 = new Float32Array([-local$$32050, -local$$32038, 0, local$$32026 - local$$32050, -local$$32038, 0, local$$32026 - local$$32050, local$$32004 - local$$32038, 0, -local$$32050, local$$32004 - local$$32038, 0]);
    this[vA] = new THREE.Vector3(-local$$32050, -local$$32038, 0);
    this[vB] = new THREE.Vector3(local$$32026 - local$$32050, -local$$32038, 0);
    this[vC] = new THREE.Vector3(local$$32026 - local$$32050, local$$32004 - local$$32038, 0);
    this[vD] = new THREE.Vector3(-local$$32050, local$$32004 - local$$32038, 0);
  } else {
    /** @type {!Float32Array} */
    local$$32066 = new Float32Array([-.5, 0, 0, local$$32026 - .5, 0, 0, local$$32026 - .5, local$$32004, 0, -.5, local$$32004, 0]);
    this[vA] = new THREE.Vector3(-.5, 0, 0);
    this[vB] = new THREE.Vector3(local$$32026, 0, 0);
    this[vC] = new THREE.Vector3(local$$32026, local$$32004, 0);
    this[vD] = new THREE.Vector3(-.5, local$$32004, 0);
  }
  /** @type {!Float32Array} */
  var local$$32174 = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  var local$$32178 = new THREE.BufferGeometry;
  local$$32178[setIndex](new THREE.BufferAttribute(local$$32001, 1));
  local$$32178[addAttribute](position, new THREE.BufferAttribute(local$$32066, 3));
  local$$32178[addAttribute](uv, new THREE.BufferAttribute(local$$32174, 2));
  this[type] = LSJBillboard;
  this[geometry] = local$$32178;
  this[material] = local$$31981 !== undefined ? local$$31981 : new THREE.SpriteMaterial({
    depthTest : false
  });
  this[geoParent] = local$$31982;
  this[camera] = undefined;
};
LSJBillboard[prototype] = Object[create](THREE[Object3D][prototype]);
/** @type {function(string, ?, string): undefined} */
LSJBillboard[prototype][constructor] = LSJBillboard;
LSJBillboard[prototype][raycast] = function() {
  var local$$32278 = new THREE.Vector3;
  return function local$$32280(local$$32281, local$$32282) {
    if (this[camera] == undefined) {
      return;
    }
    this[updateMatrixWorld]();
    var local$$32302 = new THREE.Vector3(0, 0, 0);
    var local$$32306 = new THREE.Matrix4;
    local$$32306[multiplyMatrices](this[camera][matrixWorldInverse], this[matrix]);
    var local$$32326 = local$$32302[applyMatrix4](local$$32306);
    var local$$32334 = this[camera][projectionMatrix];
    var local$$32342 = this[scale][x];
    var local$$32359 = this[vA][clone]()[multiplyScalar](local$$32342)[add](local$$32326);
    var local$$32376 = this[vB][clone]()[multiplyScalar](local$$32342)[add](local$$32326);
    var local$$32393 = this[vC][clone]()[multiplyScalar](local$$32342)[add](local$$32326);
    var local$$32410 = this[vD][clone]()[multiplyScalar](local$$32342)[add](local$$32326);
    var local$$32414 = new THREE.Vector3;
    var local$$32418 = new THREE.Matrix4;
    local$$32359 = local$$32359[applyMatrix4](this[camera][matrixWorld]);
    local$$32376 = local$$32376[applyMatrix4](this[camera][matrixWorld]);
    local$$32393 = local$$32393[applyMatrix4](this[camera][matrixWorld]);
    local$$32410 = local$$32410[applyMatrix4](this[camera][matrixWorld]);
    if (local$$32281[ray][intersectTriangle](local$$32359, local$$32376, local$$32393, false, local$$32414) != null) {
      local$$32282[push]({
        distance : local$$32414[length](),
        point : local$$32414,
        object : this[geoParent]
      });
    }
    if (local$$32281[ray][intersectTriangle](local$$32359, local$$32376, local$$32410, false, local$$32414) != null) {
      local$$32282[push]({
        distance : local$$32414[length](),
        point : local$$32414,
        object : this[geoParent]
      });
    }
  };
}();
/**
 * @return {?}
 */
LSJBillboard[prototype][getScreenRect] = function() {
  var local$$32542 = this[geoParent][getScreenRect]();
  return local$$32542;
};
/** @type {number} */
var gViewPortLeft = 0;
/** @type {number} */
var gViewPortBottom = 0;
/** @type {number} */
var gViewPortWidth = 0;
/** @type {number} */
var gViewPortHeight = 0;
var LSERangeMode = {
  RM_DISTANCE_FROM_EYE_POINT : 0,
  RM_PIXEL_SIZE_ON_SCREEN : 1
};
var LSELoadStatus = {
  LS_UNLOAD : 0,
  LS_LOADING : 1,
  LS_NET_LOADING : 2,
  LS_NET_LOADED : 3,
  LS_LOADED : 4
};
var LSEAlign = {
  TopLeft : 0,
  TopCenter : 1,
  TopRight : 2,
  MiddleLeft : 3,
  MiddleCenter : 4,
  MiddleRight : 5,
  BottomLeft : 6,
  BottomCenter : 7,
  BottomRight : 8
};
/**
 * @return {undefined}
 */
LSJUtility = function() {
};
/** @type {function(): undefined} */
LSJUtility[prototype][constructor] = LSJUtility;
/**
 * @param {?} local$$32613
 * @return {?}
 */
LSJUtility[Er] = function(local$$32613) {
  if (!LSJMath[us](local$$32613[width]) || !LSJMath[us](local$$32613[height])) {
    var local$$32639 = document[createElement](canvas);
    local$$32639[width] = LSJMath.Fs(local$$32613[width]);
    local$$32639[height] = LSJMath.Fs(local$$32613[height]);
    local$$32639[getContext](2d)[drawImage](local$$32613, 0, 0, local$$32613[width], local$$32613[height], 0, 0, local$$32639[width], local$$32639[height]);
    return local$$32639;
  }
  return local$$32613;
};
/**
 * @param {?} local$$32700
 * @return {?}
 */
LSJUtility[ensurePowerOfTwo_] = function(local$$32700) {
  if (!THREE[Math][isPowerOfTwo](local$$32700[width]) || !THREE[Math][isPowerOfTwo](local$$32700[height])) {
    var local$$32731 = document[createElement](canvas);
    local$$32731[width] = LSJMath[nextHighestPowerOfTwo_](local$$32700[width]);
    local$$32731[height] = LSJMath[nextHighestPowerOfTwo_](local$$32700[height]);
    var local$$32763 = local$$32731[getContext](2d);
    local$$32763[drawImage](local$$32700, 0, 0, local$$32700[width], local$$32700[height], 0, 0, local$$32731[width], local$$32731[height]);
    return local$$32731;
  }
  return local$$32700;
};
/**
 * @param {?} local$$32798
 * @param {?} local$$32799
 * @return {?}
 */
LSJUtility[preDealPath] = function(local$$32798, local$$32799) {
  var local$$32806 = local$$32798[substr](0);
  local$$32806[trim]();
  local$$32806[replace](\, /);
  if (local$$32806[length] >= 2) {
    if (local$$32806[substr](0, 2) == //) {
      local$$32806[0] = \;
      local$$32806[1] = \;
    }
    local$$32806[replace](//, /);
    if (local$$32806[length] > 5) {
      if (local$$32806[substr](0, 5) == http:) {
        /** @type {number} */
        var local$$32875 = 5;
        for (; local$$32875 < local$$32806[length]; local$$32875++) {
          if (local$$32806[charAt](local$$32875) != /) {
            break;
          }
        }
        var local$$32904 = local$$32806[substr](local$$32875, local$$32806[length] - local$$32875);
        local$$32806 = http:// + local$$32904;
      }
    }
  }
  if (!local$$32799) {
    /** @type {number} */
    var local$$32924 = local$$32806[length] - 1;
    for (; local$$32924 >= 0;) {
      if (local$$32806[charAt](local$$32924) != /) {
        break;
      }
      local$$32924--;
    }
    local$$32806 = local$$32806[substr](0, local$$32924 + 1);
    local$$32806 = local$$32806 + /;
  }
  return local$$32806;
};
/**
 * @param {?} local$$32971
 * @return {?}
 */
LSJUtility[getDir] = function(local$$32971) {
  var local$$32976 = local$$32971[length];
  var local$$32984 = local$$32971[lastIndexOf](/);
  if (local$$32984 < 0) {
    local$$32984 = local$$32971[lastIndexOf](\);
  }
  if (local$$32984 < 0) {
    return local$$32971[substr](0);
  }
  var local$$33016 = local$$32971[substr](0, local$$32984);
  if (local$$33016[charAt](local$$33016[length] - 1) != /) {
    local$$33016 = local$$33016 + /;
  }
  return local$$33016;
};
/**
 * @param {?} local$$33048
 * @param {?} local$$33049
 * @return {?}
 */
LSJUtility[getAbsolutePath] = function(local$$33048, local$$33049) {
  var local$$33056 = local$$33048[substr](0);
  var local$$33063 = local$$33049[substr](0);
  local$$33056[trim]();
  local$$33063[trim]();
  var local$$33075;
  if (local$$33056 == ) {
    local$$33075 = local$$33063[substr](0, 2);
    if (local$$33075 == ./ || local$$33075 == .\) {
      return local$$33063[substr](2);
    }
    local$$33075 = local$$33063[substr](0, 3);
    if (local$$33075 == ../ || local$$33075 == ..\) {
      return ;
    }
  }
  if (local$$33063 == ) {
    return local$$33063;
  }
  if (local$$33063[length] >= 2 && local$$33063[charAt](1) == : || local$$33063[length] >= 5 && local$$33063[charAt](4) == :) {
    return local$$33063;
  }
  local$$33075 = local$$33063[substr](0, 2);
  if (local$$33075 == \\ || local$$33075 == //) {
    return local$$33063;
  }
  local$$33056 = LSJUtility[preDealPath](local$$33056, false);
  local$$33063 = LSJUtility[preDealPath](local$$33063, true);
  local$$33075 = local$$33063[substr](0, 2);
  var local$$33214 = local$$33063[substr](0, 3);
  if (local$$33075 == ./) {
    local$$33063 = local$$33063[substr](2);
    return local$$33056 + local$$33063;
  } else {
    if (local$$33063[length] >= 2 && local$$33063[charAt](1) == :) {
      return local$$33063;
    } else {
      if (local$$33063[charAt](0) != / && local$$33214 != ../) {
        return local$$33056 + local$$33063;
      } else {
        if (local$$33063[charAt](0) == /) {
          return local$$33063;
        } else {
          if (local$$33214 == ../) {
            /** @type {number} */
            var local$$33274 = 0;
            do {
              local$$33274 = local$$33056[lastIndexOf](/);
              local$$33056 = local$$33056[substr](0, local$$33274);
              local$$33063 = local$$33063[substr](3);
            } while (local$$33063[substr](0, 3) == ../);
            return local$$33056 + / + local$$33063;
          }
        }
      }
    }
  }
  return local$$33048;
};
/**
 * @return {?}
 */
LSJUtility[createXMLHttp] = function() {
  /** @type {null} */
  var local$$33336 = null;
  if (window[XMLHttpRequest]) {
    /** @type {!XMLHttpRequest} */
    local$$33336 = new XMLHttpRequest;
  } else {
    local$$33336 = new ActiveXObject(Microsoft.XMLHTTP);
  }
  if (local$$33336 == null) {
    alert(您的浏览器不支持解析xml);
  }
  return local$$33336;
};
/**
 * @return {?}
 */
LSJUtility[createXMLDom] = function() {
  /** @type {!Array} */
  var local$$33386 = new Array(MSXML2.DOMDocument.5.0, MSXML2.DOMDocument.4.0, MSXML2.DOMDocument.3.0, MSXML2.DOMDocument, Microsoft.XMLDOM, MSXML.DOMDocument);
  /** @type {number} */
  var local$$33389 = 0;
  for (; local$$33389 < local$$33386[length]; local$$33389++) {
    try {
      return new ActiveXObject(local$$33386[local$$33389]);
    } catch (local$$33401) {
      return document[implementation][createDocument](, , null);
    }
  }
  return null;
};
/**
 * @param {string} local$$33429
 * @return {?}
 */
defined = function(local$$33429) {
  return local$$33429 !== undefined;
};
/**
 * @param {?} local$$33438
 * @param {!Object} local$$33439
 * @param {?} local$$33440
 * @return {?}
 */
function writeTextAndImgToCanvas(local$$33438, local$$33439, local$$33440) {
  if (local$$33438 === ) {
    return undefined;
  }
  var local$$33464 = Bold  + local$$33440[getFontSize]() + px  + local$$33440[getFontName]();
  /** @type {boolean} */
  var local$$33467 = true;
  /** @type {boolean} */
  var local$$33470 = true;
  var local$$33476 = local$$33440[getStrokeWidth]();
  var local$$33484 = document[createElement](canvas);
  /** @type {number} */
  local$$33484[width] = 1;
  /** @type {number} */
  local$$33484[height] = 1;
  local$$33484[style][font] = local$$33464;
  var local$$33512 = local$$33484[getContext](2d);
  local$$33512[font] = local$$33464;
  local$$33512[lineJoin] = round;
  local$$33512[lineWidth] = local$$33476;
  local$$33512[textBaseline] = bottom;
  local$$33484[style][visibility] = hidden;
  document[body][appendChild](local$$33484);
  var local$$33558 = measureText(local$$33512, local$$33438, local$$33467, local$$33470);
  local$$33558[computedWidth] = Math[max](local$$33558[width], local$$33558[bounds][maxx] - local$$33558[bounds][minx]);
  local$$33484[dimensions] = local$$33558;
  document[body][removeChild](local$$33484);
  local$$33484[style][visibility] = ;
  /** @type {number} */
  var local$$33609 = 0;
  if (local$$33439 != undefined) {
    local$$33609 = local$$33439[width];
    local$$33484[height] = local$$33439[height];
  } else {
    local$$33484[height] = local$$33558[height];
  }
  /** @type {number} */
  var local$$33644 = local$$33558[height] - local$$33558[ascent];
  local$$33484[width] = local$$33558[computedWidth] + local$$33609;
  /** @type {number} */
  var local$$33659 = local$$33484[height] - local$$33644;
  local$$33512[font] = local$$33464;
  local$$33512[lineJoin] = round;
  local$$33512[lineWidth] = local$$33476;
  if (local$$33467) {
    local$$33512[strokeStyle] = local$$33440[getOutlineColor]()[getStyle]();
    local$$33512[strokeText](local$$33438, local$$33609, local$$33659);
  }
  if (local$$33470) {
    local$$33512[fillStyle] = local$$33440[getFillColor]()[getStyle]();
    local$$33512[fillText](local$$33438, local$$33609, local$$33659);
  }
  return local$$33484;
}
/**
 * @param {?} local$$33724
 * @param {?} local$$33725
 * @return {?}
 */
function getCSSValue(local$$33724, local$$33725) {
  return document[defaultView][getComputedStyle](local$$33724, null)[getPropertyValue](local$$33725);
}
/**
 * @param {?} local$$33742
 * @param {?} local$$33743
 * @param {boolean} local$$33744
 * @param {boolean} local$$33745
 * @return {?}
 */
function measureText(local$$33742, local$$33743, local$$33744, local$$33745) {
  var local$$33751 = local$$33742[measureText](local$$33743);
  var local$$33759 = getCSSValue(local$$33742[canvas], font-family);
  var local$$33775 = getCSSValue(local$$33742[canvas], font-size)[replace](px, );
  /** @type {boolean} */
  var local$$33784 = !/\S/[test](local$$33743);
  local$$33751[fontsize] = local$$33775;
  var local$$33797 = document[createElement](div);
  local$$33797[style][position] = absolute;
  /** @type {number} */
  local$$33797[style][opacity] = 0;
  local$$33797[style][font] = local$$33775 + px  + local$$33759;
  local$$33797[innerHTML] = local$$33743 + <br/> + local$$33743;
  document[body][appendChild](local$$33797);
  /** @type {number} */
  local$$33751[leading] = 1.2 * local$$33775;
  var local$$33857 = getCSSValue(local$$33797, height);
  local$$33857 = local$$33857[replace](px, );
  if (local$$33857 >= local$$33775 * 2) {
    /** @type {number} */
    local$$33751[leading] = local$$33857 / 2 | 0;
  }
  document[body][removeChild](local$$33797);
  if (!local$$33784) {
    var local$$33899 = document[createElement](canvas);
    /** @type {number} */
    var local$$33902 = 100;
    local$$33899[width] = local$$33751[width] + local$$33902;
    /** @type {number} */
    local$$33899[height] = 3 * local$$33775;
    /** @type {number} */
    local$$33899[style][opacity] = 1;
    local$$33899[style][fontFamily] = local$$33759;
    local$$33899[style][fontSize] = local$$33775;
    var local$$33951 = local$$33899[getContext](2d);
    local$$33951[font] = local$$33775 + px  + local$$33759;
    var local$$33965 = local$$33899[width];
    var local$$33970 = local$$33899[height];
    /** @type {number} */
    var local$$33974 = local$$33970 / 2;
    local$$33951[fillStyle] = white;
    local$$33951[fillRect](-1, -1, local$$33965 + 2, local$$33970 + 2);
    if (local$$33744) {
      local$$33951[strokeStyle] = black;
      local$$33951[lineWidth] = local$$33742[lineWidth];
      local$$33951[strokeText](local$$33743, local$$33902 / 2, local$$33974);
    }
    if (local$$33745) {
      local$$33951[fillStyle] = black;
      local$$33951[fillText](local$$33743, local$$33902 / 2, local$$33974);
    }
    var local$$34045 = local$$33951[getImageData](0, 0, local$$33965, local$$33970)[data];
    /** @type {number} */
    var local$$34048 = 0;
    /** @type {number} */
    var local$$34052 = local$$33965 * 4;
    var local$$34057 = local$$34045[length];
    for (; ++local$$34048 < local$$34057 && local$$34045[local$$34048] === 255;) {
    }
    /** @type {number} */
    var local$$34073 = local$$34048 / local$$34052 | 0;
    /** @type {number} */
    local$$34048 = local$$34057 - 1;
    for (; --local$$34048 > 0 && local$$34045[local$$34048] === 255;) {
    }
    /** @type {number} */
    var local$$34094 = local$$34048 / local$$34052 | 0;
    /** @type {number} */
    local$$34048 = 0;
    for (; local$$34048 < local$$34057 && local$$34045[local$$34048] === 255;) {
      /** @type {number} */
      local$$34048 = local$$34048 + local$$34052;
      if (local$$34048 >= local$$34057) {
        /** @type {number} */
        local$$34048 = local$$34048 - local$$34057 + 4;
      }
    }
    /** @type {number} */
    var local$$34125 = local$$34048 % local$$34052 / 4 | 0;
    /** @type {number} */
    var local$$34128 = 1;
    /** @type {number} */
    local$$34048 = local$$34057 - 3;
    for (; local$$34048 >= 0 && local$$34045[local$$34048] === 255;) {
      /** @type {number} */
      local$$34048 = local$$34048 - local$$34052;
      if (local$$34048 < 0) {
        /** @type {number} */
        local$$34048 = local$$34057 - 3 - local$$34128++ * 4;
      }
    }
    /** @type {number} */
    var local$$34167 = local$$34048 % local$$34052 / 4 + 1 | 0;
    /** @type {number} */
    local$$33751[ascent] = local$$33974 - local$$34073;
    /** @type {number} */
    local$$33751[descent] = local$$34094 - local$$33974;
    local$$33751[bounds] = {
      minx : local$$34125 - local$$33902 / 2,
      maxx : local$$34167 - local$$33902 / 2,
      miny : 0,
      maxy : local$$34094 - local$$34073
    };
    /** @type {number} */
    local$$33751[height] = 1 + (local$$34094 - local$$34073);
  } else {
    /** @type {number} */
    local$$33751[ascent] = 0;
    /** @type {number} */
    local$$33751[descent] = 0;
    local$$33751[bounds] = {
      minx : 0,
      maxx : local$$33751[width],
      miny : 0,
      maxy : 0
    };
    /** @type {number} */
    local$$33751[height] = 0;
  }
  return local$$33751;
}
/**
 * @param {number} local$$34243
 * @param {number} local$$34244
 * @param {number} local$$34245
 * @param {number} local$$34246
 * @return {?}
 */
function projectedRadius(local$$34243, local$$34244, local$$34245, local$$34246) {
  /** @type {number} */
  var local$$34257 = 1 / Math[tan](local$$34244 / 2) / local$$34245;
  /** @type {number} */
  local$$34257 = local$$34257 * local$$34246 / 2;
  return local$$34243 * local$$34257;
}
/**
 * @param {?} local$$34273
 * @param {?} local$$34274
 * @param {?} local$$34275
 * @return {undefined}
 */
THREE[DragControls] = function(local$$34273, local$$34274, local$$34275) {
  /**
   * @param {?} local$$34278
   * @return {undefined}
   */
  function local$$34277(local$$34278) {
    local$$34278[preventDefault]();
    /** @type {number} */
    local$$34285[x] = local$$34278[clientX] / local$$34275[width] * 2 - 1;
    /** @type {number} */
    local$$34285[y] = -(local$$34278[clientY] / local$$34275[height]) * 2 + 1;
    local$$34319[setFromCamera](local$$34285, local$$34273);
    var local$$34328 = local$$34319[ray];
    if (local$$34330 && local$$34331[enabled]) {
      var local$$34339 = local$$34330[normal];
      var local$$34348 = local$$34339[dot](local$$34328[direction]);
      if (local$$34348 == 0) {
        console[log](no or infinite solutions);
        return;
      }
      var local$$34382 = local$$34339[dot](local$$34366[copy](local$$34330[point])[sub](local$$34328[origin]));
      /** @type {number} */
      var local$$34385 = local$$34382 / local$$34348;
      local$$34387[copy](local$$34328[direction])[multiplyScalar](local$$34385)[add](local$$34328[origin])[sub](local$$34409);
      var local$$34412;
      var local$$34414;
      /** @type {boolean} */
      var local$$34417 = false;
      var local$$34419;
      var local$$34421;
      var local$$34423;
      if (local$$34412) {
        /** @type {boolean} */
        local$$34419 = true;
        /** @type {boolean} */
        local$$34421 = false;
        /** @type {boolean} */
        local$$34423 = false;
      } else {
        if (local$$34414) {
          /** @type {boolean} */
          local$$34419 = false;
          /** @type {boolean} */
          local$$34421 = true;
          /** @type {boolean} */
          local$$34423 = false;
        } else {
          /** @type {boolean} */
          local$$34419 = local$$34421 = local$$34423 = true;
        }
      }
      if (local$$34419) {
        local$$34330[object][position][x] = local$$34387[x];
      }
      if (local$$34421) {
        local$$34330[object][position][y] = local$$34387[y];
      }
      if (local$$34423) {
        local$$34330[object][position][z] = local$$34387[z];
      }
      local$$34506(drag, local$$34330);
      return;
    }
    local$$34319[setFromCamera](local$$34285, local$$34273);
    var local$$34524 = local$$34319[intersectObjects](local$$34274);
    if (local$$34524[length] > 0) {
      local$$34275[style][cursor] = pointer;
      local$$34541 = local$$34524[0];
      local$$34506(hoveron, local$$34541);
    } else {
      local$$34506(hoveroff, local$$34541);
      /** @type {null} */
      local$$34541 = null;
      local$$34275[style][cursor] = auto;
    }
  }
  /**
   * @param {?} local$$34573
   * @return {undefined}
   */
  function local$$34572(local$$34573) {
    local$$34573[preventDefault]();
    /** @type {number} */
    local$$34285[x] = local$$34573[clientX] / local$$34275[width] * 2 - 1;
    /** @type {number} */
    local$$34285[y] = -(local$$34573[clientY] / local$$34275[height]) * 2 + 1;
    local$$34319[setFromCamera](local$$34285, local$$34273);
    var local$$34622 = local$$34319[intersectObjects](local$$34274);
    var local$$34627 = local$$34319[ray];
    var local$$34632 = local$$34627[direction];
    if (local$$34622[length] > 0) {
      local$$34330 = local$$34622[0];
      local$$34330[ray] = local$$34627;
      local$$34330[normal] = local$$34632;
      local$$34409[copy](local$$34330[point])[sub](local$$34330[object][position]);
      local$$34275[style][cursor] = move;
      local$$34506(dragstart, local$$34330);
    }
  }
  /**
   * @param {?} local$$34690
   * @return {undefined}
   */
  function local$$34689(local$$34690) {
    local$$34690[preventDefault]();
    if (local$$34330) {
      local$$34506(dragend, local$$34330);
      /** @type {null} */
      local$$34330 = null;
    }
    local$$34275[style][cursor] = auto;
  }
  var local$$34721 = new THREE.Projector;
  var local$$34319 = new THREE.Raycaster;
  var local$$34285 = new THREE.Vector3;
  var local$$34409 = new THREE.Vector3;
  var local$$34330;
  var local$$34541;
  var local$$34366 = new THREE.Vector3;
  var local$$34387 = new THREE.Vector3;
  var local$$34743 = new THREE.Vector3;
  /** @type {boolean} */
  this[enabled] = false;
  var local$$34752 = {};
  var local$$34331 = this;
  /**
   * @param {?} local$$34758
   * @param {?} local$$34759
   * @return {?}
   */
  this[on] = function(local$$34758, local$$34759) {
    if (!local$$34752[local$$34758]) {
      /** @type {!Array} */
      local$$34752[local$$34758] = [];
    }
    local$$34752[local$$34758][push](local$$34759);
    return local$$34331;
  };
  /**
   * @param {?} local$$34784
   * @param {?} local$$34785
   * @return {?}
   */
  this[off] = function(local$$34784, local$$34785) {
    var local$$34788 = local$$34752[local$$34784];
    if (!local$$34788) {
      return local$$34331;
    }
    if (local$$34788[indexOf](local$$34785) > -1) {
      local$$34788[splice](local$$34785, 1);
    }
    return local$$34331;
  };
  /**
   * @param {?} local$$34815
   * @param {?} local$$34816
   * @param {?} local$$34817
   * @return {undefined}
   */
  var local$$34506 = function(local$$34815, local$$34816, local$$34817) {
    var local$$34820 = local$$34752[local$$34815];
    if (!local$$34820) {
      return;
    }
    if (!local$$34817) {
      /** @type {number} */
      var local$$34829 = 0;
      for (; local$$34829 < local$$34820[length]; local$$34829++) {
        local$$34820[local$$34829](local$$34816);
      }
    }
  };
  /**
   * @param {?} local$$34850
   * @return {undefined}
   */
  this[setObjects] = function(local$$34850) {
    if (local$$34850 instanceof THREE[Scene]) {
      local$$34274 = local$$34850[children];
    } else {
      local$$34274 = local$$34850;
    }
  };
  this[setObjects](local$$34274);
  /**
   * @return {undefined}
   */
  this[activate] = function() {
    local$$34275[addEventListener](mousemove, local$$34277, false);
    local$$34275[addEventListener](mousedown, local$$34572, false);
    local$$34275[addEventListener](mouseup, local$$34689, false);
  };
  /**
   * @return {undefined}
   */
  this[deactivate] = function() {
    local$$34275[removeEventListener](mousemove, local$$34277, false);
    local$$34275[removeEventListener](mousedown, local$$34572, false);
    local$$34275[removeEventListener](mouseup, local$$34689, false);
  };
  /**
   * @return {undefined}
   */
  this[dispose] = function() {
    local$$34331[deactivate]();
  };
  this[activate]();
};
/**
 * @return {undefined}
 */
LSJLayers = function() {
  /** @type {!Array} */
  this[layers] = [];
  this[meshGroup] = new THREE.Group;
  this[boundingSphere] = new THREE.Sphere;
};
/** @type {function(): undefined} */
LSJLayers[prototype][constructor] = LSJLayers;
/**
 * @return {undefined}
 */
LSJLayers[prototype][dispose] = function() {
  var local$$35009 = this[layers][length];
  /** @type {number} */
  var local$$35012 = 0;
  for (; local$$35012 < local$$35009; local$$35012++) {
    var local$$35021 = this[layers][local$$35012];
    this[meshGroup][remove](local$$35021[meshGroup]);
    if (local$$35021 != null) {
      local$$35021[dispose]();
    }
  }
  this[layers][slice](0, local$$35009);
};
/**
 * @param {!Object} local$$35065
 * @return {undefined}
 */
LSJLayers[prototype][addLayer] = function(local$$35065) {
  if (local$$35065 == null || local$$35065 == undefined) {
    return;
  }
  this[layers][push](local$$35065);
  this[meshGroup][add](local$$35065[meshGroup]);
  LSJMath[expandSphere](this[boundingSphere], local$$35065[getBoundingSphere]());
};
/**
 * @param {?} local$$35116
 * @return {?}
 */
LSJLayers[prototype][getLayerByCaption] = function(local$$35116) {
  var local$$35124 = this[layers][length];
  /** @type {number} */
  var local$$35127 = 0;
  for (; local$$35127 < local$$35124; local$$35127++) {
    var local$$35136 = this[layers][local$$35127];
    if (local$$35136[caption] == local$$35116) {
      return local$$35136;
    }
  }
  return null;
};
/**
 * @param {?} local$$35160
 * @return {?}
 */
LSJLayers[prototype][getLayerByName] = function(local$$35160) {
  var local$$35168 = this[layers][length];
  /** @type {number} */
  var local$$35171 = 0;
  for (; local$$35171 < local$$35168; local$$35171++) {
    var local$$35180 = local$$35160[toLowerCase]();
    var local$$35186 = this[layers][local$$35171];
    if (local$$35186 != null) {
      var local$$35197 = local$$35186[name][toLowerCase]();
      if (local$$35197 == local$$35180) {
        return local$$35186;
      }
    }
  }
  return null;
};
/**
 * @param {number} local$$35220
 * @return {?}
 */
LSJLayers[prototype][getLayerByIndex] = function(local$$35220) {
  var local$$35228 = this[layers][length];
  if (local$$35220 >= 0 && local$$35220 < local$$35228) {
    return this[layers][local$$35220];
  }
  return null;
};
/**
 * @return {?}
 */
LSJLayers[prototype][getBoundingSphere] = function() {
  if (this[boundingSphere][empty]()) {
    var local$$35268 = this[layers][length];
    /** @type {number} */
    var local$$35271 = 0;
    for (; local$$35271 < local$$35268; local$$35271++) {
      var local$$35280 = this[layers][local$$35271];
      if (local$$35280 != null) {
        LSJMath[expandSphere](this[boundingSphere], local$$35280[getBoundingSphere]());
      }
    }
  }
  return this[boundingSphere];
};
/**
 * @return {undefined}
 */
LSJLayers[prototype][releaseSelection] = function() {
  var local$$35324 = this[layers][length];
  /** @type {number} */
  var local$$35327 = 0;
  for (; local$$35327 < local$$35324; local$$35327++) {
    var local$$35336 = this[layers][local$$35327];
    if (local$$35336 != null && local$$35336[type] == FeatureLayer) {
      local$$35336[releaseSelection]();
    }
  }
};
/**
 * @param {?} local$$35366
 * @return {undefined}
 */
LSJLayers[prototype][render] = function(local$$35366) {
  var local$$35374 = this[layers][length];
  /** @type {number} */
  var local$$35377 = 0;
  for (; local$$35377 < local$$35374; local$$35377++) {
    var local$$35386 = this[layers][local$$35377];
    if (local$$35386 != null) {
      local$$35386[render](local$$35366);
    }
  }
};
/**
 * @return {undefined}
 */
THREE[RenderableObject] = function() {
  /** @type {number} */
  this[id] = 0;
  /** @type {null} */
  this[object] = null;
  /** @type {number} */
  this[z] = 0;
  /** @type {number} */
  this[renderOrder] = 0;
};
/**
 * @return {undefined}
 */
THREE[RenderableFace] = function() {
  /** @type {number} */
  this[id] = 0;
  this[v1] = new THREE.RenderableVertex;
  this[v2] = new THREE.RenderableVertex;
  this[v3] = new THREE.RenderableVertex;
  this[normalModel] = new THREE.Vector3;
  /** @type {!Array} */
  this[vertexNormalsModel] = [new THREE.Vector3, new THREE.Vector3, new THREE.Vector3];
  /** @type {number} */
  this[vertexNormalsLength] = 0;
  this[color] = new THREE.Color;
  /** @type {null} */
  this[material] = null;
  /** @type {!Array} */
  this[uvs] = [new THREE.Vector2, new THREE.Vector2, new THREE.Vector2];
  /** @type {number} */
  this[z] = 0;
  /** @type {number} */
  this[renderOrder] = 0;
};
/**
 * @return {undefined}
 */
THREE[RenderableVertex] = function() {
  this[position] = new THREE.Vector3;
  this[positionWorld] = new THREE.Vector3;
  this[positionScreen] = new THREE.Vector4;
  /** @type {boolean} */
  this[visible] = true;
};
/**
 * @param {?} local$$35579
 * @return {undefined}
 */
THREE[RenderableVertex][prototype][copy] = function(local$$35579) {
  this[positionWorld][copy](local$$35579[positionWorld]);
  this[positionScreen][copy](local$$35579[positionScreen]);
};
/**
 * @return {undefined}
 */
THREE[RenderableLine] = function() {
  /** @type {number} */
  this[id] = 0;
  this[v1] = new THREE.RenderableVertex;
  this[v2] = new THREE.RenderableVertex;
  /** @type {!Array} */
  this[vertexColors] = [new THREE.Color, new THREE.Color];
  /** @type {null} */
  this[material] = null;
  /** @type {number} */
  this[z] = 0;
  /** @type {number} */
  this[renderOrder] = 0;
};
/**
 * @return {undefined}
 */
THREE[RenderableSprite] = function() {
  /** @type {number} */
  this[id] = 0;
  /** @type {null} */
  this[object] = null;
  /** @type {number} */
  this[x] = 0;
  /** @type {number} */
  this[y] = 0;
  /** @type {number} */
  this[z] = 0;
  /** @type {number} */
  this[rotation] = 0;
  this[scale] = new THREE.Vector2;
  /** @type {null} */
  this[material] = null;
  /** @type {number} */
  this[renderOrder] = 0;
};
/**
 * @return {undefined}
 */
THREE[Projector] = function() {
  /**
   * @return {?}
   */
  function local$$35730() {
    if (local$$35732 === local$$35733) {
      var local$$35737 = new THREE.RenderableObject;
      local$$35739[push](local$$35737);
      local$$35733++;
      local$$35732++;
      return local$$35737;
    }
    return local$$35739[local$$35732++];
  }
  /**
   * @return {?}
   */
  function local$$35758() {
    if (local$$35760 === local$$35761) {
      var local$$35765 = new THREE.RenderableVertex;
      local$$35767[push](local$$35765);
      local$$35761++;
      local$$35760++;
      return local$$35765;
    }
    return local$$35767[local$$35760++];
  }
  /**
   * @return {?}
   */
  function local$$35786() {
    if (local$$35788 === local$$35789) {
      var local$$35793 = new THREE.RenderableFace;
      local$$35795[push](local$$35793);
      local$$35789++;
      local$$35788++;
      return local$$35793;
    }
    return local$$35795[local$$35788++];
  }
  /**
   * @return {?}
   */
  function local$$35814() {
    if (local$$35816 === local$$35817) {
      var local$$35821 = new THREE.RenderableLine;
      local$$35823[push](local$$35821);
      local$$35817++;
      local$$35816++;
      return local$$35821;
    }
    return local$$35823[local$$35816++];
  }
  /**
   * @return {?}
   */
  function local$$35842() {
    if (local$$35844 === local$$35845) {
      var local$$35849 = new THREE.RenderableSprite;
      local$$35851[push](local$$35849);
      local$$35845++;
      local$$35844++;
      return local$$35849;
    }
    return local$$35851[local$$35844++];
  }
  /**
   * @param {?} local$$35871
   * @param {?} local$$35872
   * @return {?}
   */
  function local$$35870(local$$35871, local$$35872) {
    if (local$$35871[renderOrder] !== local$$35872[renderOrder]) {
      return local$$35871[renderOrder] - local$$35872[renderOrder];
    } else {
      if (local$$35871[z] !== local$$35872[z]) {
        return local$$35872[z] - local$$35871[z];
      } else {
        if (local$$35871[id] !== local$$35872[id]) {
          return local$$35871[id] - local$$35872[id];
        } else {
          return 0;
        }
      }
    }
  }
  /**
   * @param {?} local$$35933
   * @param {?} local$$35934
   * @return {?}
   */
  function local$$35932(local$$35933, local$$35934) {
    /** @type {number} */
    var local$$35937 = 0;
    /** @type {number} */
    var local$$35940 = 1;
    var local$$35949 = local$$35933[z] + local$$35933[w];
    var local$$35958 = local$$35934[z] + local$$35934[w];
    var local$$35968 = -local$$35933[z] + local$$35933[w];
    var local$$35978 = -local$$35934[z] + local$$35934[w];
    if (local$$35949 >= 0 && local$$35958 >= 0 && local$$35968 >= 0 && local$$35978 >= 0) {
      return true;
    } else {
      if (local$$35949 < 0 && local$$35958 < 0 || local$$35968 < 0 && local$$35978 < 0) {
        return false;
      } else {
        if (local$$35949 < 0) {
          local$$35937 = Math[max](local$$35937, local$$35949 / (local$$35949 - local$$35958));
        } else {
          if (local$$35958 < 0) {
            local$$35940 = Math[min](local$$35940, local$$35949 / (local$$35949 - local$$35958));
          }
        }
        if (local$$35968 < 0) {
          local$$35937 = Math[max](local$$35937, local$$35968 / (local$$35968 - local$$35978));
        } else {
          if (local$$35978 < 0) {
            local$$35940 = Math[min](local$$35940, local$$35968 / (local$$35968 - local$$35978));
          }
        }
        if (local$$35940 < local$$35937) {
          return false;
        } else {
          local$$35933[lerp](local$$35934, local$$35937);
          local$$35934[lerp](local$$35933, 1 - local$$35940);
          return true;
        }
      }
    }
  }
  var local$$36086;
  var local$$35732;
  /** @type {!Array} */
  var local$$35739 = [];
  /** @type {number} */
  var local$$35733 = 0;
  var local$$36093;
  var local$$35760;
  /** @type {!Array} */
  var local$$35767 = [];
  /** @type {number} */
  var local$$35761 = 0;
  var local$$36100;
  var local$$35788;
  /** @type {!Array} */
  var local$$35795 = [];
  /** @type {number} */
  var local$$35789 = 0;
  var local$$36107;
  var local$$35816;
  /** @type {!Array} */
  var local$$35823 = [];
  /** @type {number} */
  var local$$35817 = 0;
  var local$$36114;
  var local$$35844;
  /** @type {!Array} */
  var local$$35851 = [];
  /** @type {number} */
  var local$$35845 = 0;
  var local$$36125 = {
    objects : [],
    lights : [],
    elements : []
  };
  var local$$36129 = new THREE.Vector3;
  var local$$36133 = new THREE.Vector4;
  var local$$36148 = new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  var local$$36152 = new THREE.Box3;
  /** @type {!Array} */
  var local$$36156 = new Array(3);
  /** @type {!Array} */
  var local$$36160 = new Array(4);
  var local$$36164 = new THREE.Matrix4;
  var local$$36168 = new THREE.Matrix4;
  var local$$36170;
  var local$$36174 = new THREE.Matrix4;
  var local$$36179 = new THREE.Matrix3;
  var local$$36184 = new THREE.Frustum;
  var local$$36188 = new THREE.Vector4;
  var local$$36192 = new THREE.Vector4;
  /**
   * @param {?} local$$36197
   * @param {?} local$$36198
   * @return {undefined}
   */
  this[projectVector] = function(local$$36197, local$$36198) {
    console[warn](THREE.Projector: .projectVector() is now vector.project().);
    local$$36197[project](local$$36198);
  };
  /**
   * @param {?} local$$36219
   * @param {?} local$$36220
   * @return {undefined}
   */
  this[unprojectVector] = function(local$$36219, local$$36220) {
    console[warn](THREE.Projector: .unprojectVector() is now vector.unproject().);
    local$$36219[unproject](local$$36220);
  };
  /**
   * @param {?} local$$36241
   * @param {?} local$$36242
   * @return {undefined}
   */
  this[pickingRay] = function(local$$36241, local$$36242) {
    console[error](THREE.Projector: .pickingRay() is now raycaster.setFromCamera().);
  };
  /**
   * @return {?}
   */
  var local$$36865 = function() {
    /** @type {!Array} */
    var local$$36257 = [];
    /** @type {!Array} */
    var local$$36260 = [];
    /** @type {null} */
    var local$$36263 = null;
    /** @type {null} */
    var local$$36266 = null;
    var local$$36270 = new THREE.Matrix3;
    /**
     * @param {!Object} local$$36272
     * @return {undefined}
     */
    var local$$36303 = function(local$$36272) {
      /** @type {!Object} */
      local$$36263 = local$$36272;
      local$$36266 = local$$36263[material];
      local$$36270[getNormalMatrix](local$$36263[matrixWorld]);
      /** @type {number} */
      local$$36257[length] = 0;
      /** @type {number} */
      local$$36260[length] = 0;
    };
    /**
     * @param {?} local$$36305
     * @return {undefined}
     */
    var local$$36404 = function(local$$36305) {
      var local$$36310 = local$$36305[position];
      var local$$36315 = local$$36305[positionWorld];
      var local$$36320 = local$$36305[positionScreen];
      local$$36315[copy](local$$36310)[applyMatrix4](local$$36170);
      local$$36320[copy](local$$36315)[applyMatrix4](local$$36168);
      /** @type {number} */
      var local$$36345 = 1 / local$$36320[w];
      local$$36320[x] *= local$$36345;
      local$$36320[y] *= local$$36345;
      local$$36320[z] *= local$$36345;
      /** @type {boolean} */
      local$$36305[visible] = local$$36320[x] >= -1 && local$$36320[x] <= 1 && local$$36320[y] >= -1 && local$$36320[y] <= 1 && local$$36320[z] >= -1 && local$$36320[z] <= 1;
    };
    /**
     * @param {?} local$$36406
     * @param {?} local$$36407
     * @param {?} local$$36408
     * @return {undefined}
     */
    var local$$36425 = function(local$$36406, local$$36407, local$$36408) {
      local$$36093 = local$$35758();
      local$$36093[position][set](local$$36406, local$$36407, local$$36408);
      local$$36404(local$$36093);
    };
    /**
     * @param {?} local$$36427
     * @param {?} local$$36428
     * @param {?} local$$36429
     * @return {undefined}
     */
    var local$$36438 = function(local$$36427, local$$36428, local$$36429) {
      local$$36257[push](local$$36427, local$$36428, local$$36429);
    };
    /**
     * @param {?} local$$36440
     * @param {?} local$$36441
     * @return {undefined}
     */
    var local$$36450 = function(local$$36440, local$$36441) {
      local$$36260[push](local$$36440, local$$36441);
    };
    /**
     * @param {?} local$$36452
     * @param {?} local$$36453
     * @param {?} local$$36454
     * @return {?}
     */
    var local$$36510 = function(local$$36452, local$$36453, local$$36454) {
      if (local$$36452[visible] === true || local$$36453[visible] === true || local$$36454[visible] === true) {
        return true;
      }
      local$$36156[0] = local$$36452[positionScreen];
      local$$36156[1] = local$$36453[positionScreen];
      local$$36156[2] = local$$36454[positionScreen];
      return local$$36148[isIntersectionBox](local$$36152[setFromPoints](local$$36156));
    };
    /**
     * @param {?} local$$36512
     * @param {?} local$$36513
     * @param {?} local$$36514
     * @return {?}
     */
    var local$$36576 = function(local$$36512, local$$36513, local$$36514) {
      return (local$$36514[positionScreen][x] - local$$36512[positionScreen][x]) * (local$$36513[positionScreen][y] - local$$36512[positionScreen][y]) - (local$$36514[positionScreen][y] - local$$36512[positionScreen][y]) * (local$$36513[positionScreen][x] - local$$36512[positionScreen][x]) < 0;
    };
    /**
     * @param {undefined} local$$36578
     * @param {undefined} local$$36579
     * @return {undefined}
     */
    var local$$36660 = function(local$$36578, local$$36579) {
      var local$$36582 = local$$35767[local$$36578];
      var local$$36585 = local$$35767[local$$36579];
      local$$36107 = local$$35814();
      local$$36107[id] = local$$36263[id];
      local$$36107[v1][copy](local$$36582);
      local$$36107[v2][copy](local$$36585);
      /** @type {number} */
      local$$36107[z] = (local$$36582[positionScreen][z] + local$$36585[positionScreen][z]) / 2;
      local$$36107[renderOrder] = local$$36263[renderOrder];
      local$$36107[material] = local$$36263[material];
      local$$36125[elements][push](local$$36107);
    };
    /**
     * @param {number} local$$36662
     * @param {undefined} local$$36663
     * @param {undefined} local$$36664
     * @return {undefined}
     */
    var local$$36858 = function(local$$36662, local$$36663, local$$36664) {
      var local$$36667 = local$$35767[local$$36662];
      var local$$36670 = local$$35767[local$$36663];
      var local$$36673 = local$$35767[local$$36664];
      if (local$$36510(local$$36667, local$$36670, local$$36673) === false) {
        return;
      }
      if (local$$36266[side] === THREE[DoubleSide] || local$$36576(local$$36667, local$$36670, local$$36673) === true) {
        local$$36100 = local$$35786();
        local$$36100[id] = local$$36263[id];
        local$$36100[v1][copy](local$$36667);
        local$$36100[v2][copy](local$$36670);
        local$$36100[v3][copy](local$$36673);
        /** @type {number} */
        local$$36100[z] = (local$$36667[positionScreen][z] + local$$36670[positionScreen][z] + local$$36673[positionScreen][z]) / 3;
        local$$36100[renderOrder] = local$$36263[renderOrder];
        local$$36100[normalModel][fromArray](local$$36257, local$$36662 * 3);
        local$$36100[normalModel][applyMatrix3](local$$36270)[normalize]();
        /** @type {number} */
        var local$$36786 = 0;
        for (; local$$36786 < 3; local$$36786++) {
          var local$$36796 = local$$36100[vertexNormalsModel][local$$36786];
          local$$36796[fromArray](local$$36257, arguments[local$$36786] * 3);
          local$$36796[applyMatrix3](local$$36270)[normalize]();
          var local$$36819 = local$$36100[uvs][local$$36786];
          local$$36819[fromArray](local$$36260, arguments[local$$36786] * 2);
        }
        /** @type {number} */
        local$$36100[vertexNormalsLength] = 3;
        local$$36100[material] = local$$36263[material];
        local$$36125[elements][push](local$$36100);
      }
    };
    return {
      setObject : local$$36303,
      projectVertex : local$$36404,
      checkTriangleVisibility : local$$36510,
      checkBackfaceCulling : local$$36576,
      pushVertex : local$$36425,
      pushNormal : local$$36438,
      pushUv : local$$36450,
      pushLine : local$$36660,
      pushTriangle : local$$36858
    };
  };
  var local$$36868 = new local$$36865;
  /**
   * @param {?} local$$36873
   * @param {?} local$$36874
   * @param {boolean} local$$36875
   * @param {boolean} local$$36876
   * @return {?}
   */
  this[projectScene] = function(local$$36873, local$$36874, local$$36875, local$$36876) {
    /** @type {number} */
    local$$35788 = 0;
    /** @type {number} */
    local$$35816 = 0;
    /** @type {number} */
    local$$35844 = 0;
    /** @type {number} */
    local$$36125[elements][length] = 0;
    if (local$$36873[autoUpdate] === true) {
      local$$36873[updateMatrixWorld]();
    }
    if (local$$36874[parent] === null) {
      local$$36874[updateMatrixWorld]();
    }
    local$$36164[copy](local$$36874[matrixWorldInverse][getInverse](local$$36874[matrixWorld]));
    local$$36168[multiplyMatrices](local$$36874[projectionMatrix], local$$36164);
    local$$36184[setFromMatrix](local$$36168);
    /** @type {number} */
    local$$35732 = 0;
    /** @type {number} */
    local$$36125[objects][length] = 0;
    /** @type {number} */
    local$$36125[lights][length] = 0;
    local$$36873[traverseVisible](function(local$$36974) {
      if (local$$36974 instanceof THREE[Light]) {
        local$$36125[lights][push](local$$36974);
      } else {
        if (local$$36974 instanceof THREE[Mesh] || local$$36974 instanceof THREE[Line] || local$$36974 instanceof THREE[Sprite]) {
          var local$$37006 = local$$36974[material];
          if (local$$37006[visible] === false) {
            return;
          }
          if (local$$36974[frustumCulled] === false || local$$36184[intersectsObject](local$$36974) === true) {
            local$$36086 = local$$35730();
            local$$36086[id] = local$$36974[id];
            local$$36086[object] = local$$36974;
            local$$36129[setFromMatrixPosition](local$$36974[matrixWorld]);
            local$$36129[applyProjection](local$$36168);
            local$$36086[z] = local$$36129[z];
            local$$36086[renderOrder] = local$$36974[renderOrder];
            local$$36125[objects][push](local$$36086);
          }
        }
      }
    });
    if (local$$36875 === true) {
      local$$36125[objects][sort](local$$35870);
    }
    /** @type {number} */
    var local$$37106 = 0;
    var local$$37114 = local$$36125[objects][length];
    for (; local$$37106 < local$$37114; local$$37106++) {
      var local$$37126 = local$$36125[objects][local$$37106][object];
      var local$$37131 = local$$37126[geometry];
      local$$36868[setObject](local$$37126);
      local$$36170 = local$$37126[matrixWorld];
      /** @type {number} */
      local$$35760 = 0;
      if (local$$37126 instanceof THREE[Mesh]) {
        if (local$$37131 instanceof THREE[BufferGeometry]) {
          var local$$37157 = local$$37131[attributes];
          var local$$37162 = local$$37131[groups];
          if (local$$37157[position] === undefined) {
            continue;
          }
          var local$$37178 = local$$37157[position][array];
          /** @type {number} */
          var local$$37181 = 0;
          var local$$37186 = local$$37178[length];
          for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
            local$$36868[pushVertex](local$$37178[local$$37181], local$$37178[local$$37181 + 1], local$$37178[local$$37181 + 2]);
          }
          if (local$$37157[normal] !== undefined) {
            var local$$37218 = local$$37157[normal][array];
            /** @type {number} */
            local$$37181 = 0;
            local$$37186 = local$$37218[length];
            for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
              local$$36868[pushNormal](local$$37218[local$$37181], local$$37218[local$$37181 + 1], local$$37218[local$$37181 + 2]);
            }
          }
          if (local$$37157[uv] !== undefined) {
            var local$$37260 = local$$37157[uv][array];
            /** @type {number} */
            local$$37181 = 0;
            local$$37186 = local$$37260[length];
            for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 2) {
              local$$36868[pushUv](local$$37260[local$$37181], local$$37260[local$$37181 + 1]);
            }
          }
          if (local$$37131[index] !== null) {
            var local$$37300 = local$$37131[index][array];
            if (local$$37162[length] > 0) {
              /** @type {number} */
              local$$37106 = 0;
              for (; local$$37106 < local$$37162[length]; local$$37106++) {
                var local$$37317 = local$$37162[local$$37106];
                local$$37181 = local$$37317[start];
                local$$37186 = local$$37317[start] + local$$37317[count];
                for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
                  local$$36868[pushTriangle](local$$37300[local$$37181], local$$37300[local$$37181 + 1], local$$37300[local$$37181 + 2]);
                }
              }
            } else {
              /** @type {number} */
              local$$37181 = 0;
              local$$37186 = local$$37300[length];
              for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
                local$$36868[pushTriangle](local$$37300[local$$37181], local$$37300[local$$37181 + 1], local$$37300[local$$37181 + 2]);
              }
            }
          } else {
            /** @type {number} */
            local$$37181 = 0;
            /** @type {number} */
            local$$37186 = local$$37178[length] / 3;
            for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
              local$$36868[pushTriangle](local$$37181, local$$37181 + 1, local$$37181 + 2);
            }
          }
        } else {
          if (local$$37131 instanceof THREE[Geometry]) {
            var local$$37421 = local$$37131[vertices];
            var local$$37426 = local$$37131[faces];
            var local$$37433 = local$$37131[faceVertexUvs][0];
            local$$36179[getNormalMatrix](local$$36170);
            var local$$37443 = local$$37126[material];
            /** @type {boolean} */
            var local$$37449 = local$$37443 instanceof THREE[MeshFaceMaterial];
            var local$$37458 = local$$37449 === true ? local$$37126[material] : null;
            /** @type {number} */
            var local$$37461 = 0;
            var local$$37466 = local$$37421[length];
            for (; local$$37461 < local$$37466; local$$37461++) {
              var local$$37472 = local$$37421[local$$37461];
              local$$36129[copy](local$$37472);
              if (local$$37443[morphTargets] === true) {
                var local$$37487 = local$$37131[morphTargets];
                var local$$37492 = local$$37126[morphTargetInfluences];
                /** @type {number} */
                var local$$37495 = 0;
                var local$$37500 = local$$37487[length];
                for (; local$$37495 < local$$37500; local$$37495++) {
                  var local$$37506 = local$$37492[local$$37495];
                  if (local$$37506 === 0) {
                    continue;
                  }
                  var local$$37515 = local$$37487[local$$37495];
                  var local$$37521 = local$$37515[vertices][local$$37461];
                  local$$36129[x] += (local$$37521[x] - local$$37472[x]) * local$$37506;
                  local$$36129[y] += (local$$37521[y] - local$$37472[y]) * local$$37506;
                  local$$36129[z] += (local$$37521[z] - local$$37472[z]) * local$$37506;
                }
              }
              local$$36868[pushVertex](local$$36129[x], local$$36129[y], local$$36129[z]);
            }
            /** @type {number} */
            var local$$37585 = 0;
            var local$$37590 = local$$37426[length];
            for (; local$$37585 < local$$37590; local$$37585++) {
              var local$$37596 = local$$37426[local$$37585];
              local$$37443 = local$$37449 === true ? local$$37458[materials][local$$37596[materialIndex]] : local$$37126[material];
              if (local$$37443 === undefined) {
                continue;
              }
              var local$$37621 = local$$37443[side];
              var local$$37627 = local$$35767[local$$37596[a]];
              var local$$37633 = local$$35767[local$$37596[b]];
              var local$$37639 = local$$35767[local$$37596[c]];
              if (local$$36868[checkTriangleVisibility](local$$37627, local$$37633, local$$37639) === false) {
                continue;
              }
              var local$$37655 = local$$36868[checkBackfaceCulling](local$$37627, local$$37633, local$$37639);
              if (local$$37621 !== THREE[DoubleSide]) {
                if (local$$37621 === THREE[FrontSide] && local$$37655 === false) {
                  continue;
                }
                if (local$$37621 === THREE[BackSide] && local$$37655 === true) {
                  continue;
                }
              }
              local$$36100 = local$$35786();
              local$$36100[id] = local$$37126[id];
              local$$36100[v1][copy](local$$37627);
              local$$36100[v2][copy](local$$37633);
              local$$36100[v3][copy](local$$37639);
              local$$36100[normalModel][copy](local$$37596[normal]);
              if (local$$37655 === false && (local$$37621 === THREE[BackSide] || local$$37621 === THREE[DoubleSide])) {
                local$$36100[normalModel][negate]();
              }
              local$$36100[normalModel][applyMatrix3](local$$36179)[normalize]();
              var local$$37769 = local$$37596[vertexNormals];
              /** @type {number} */
              var local$$37772 = 0;
              var local$$37782 = Math[min](local$$37769[length], 3);
              for (; local$$37772 < local$$37782; local$$37772++) {
                var local$$37791 = local$$36100[vertexNormalsModel][local$$37772];
                local$$37791[copy](local$$37769[local$$37772]);
                if (local$$37655 === false && (local$$37621 === THREE[BackSide] || local$$37621 === THREE[DoubleSide])) {
                  local$$37791[negate]();
                }
                local$$37791[applyMatrix3](local$$36179)[normalize]();
              }
              local$$36100[vertexNormalsLength] = local$$37769[length];
              var local$$37840 = local$$37433[local$$37585];
              if (local$$37840 !== undefined) {
                /** @type {number} */
                var local$$37844 = 0;
                for (; local$$37844 < 3; local$$37844++) {
                  local$$36100[uvs][local$$37844][copy](local$$37840[local$$37844]);
                }
              }
              local$$36100[color] = local$$37596[color];
              local$$36100[material] = local$$37443;
              /** @type {number} */
              local$$36100[z] = (local$$37627[positionScreen][z] + local$$37633[positionScreen][z] + local$$37639[positionScreen][z]) / 3;
              local$$36100[renderOrder] = local$$37126[renderOrder];
              local$$36125[elements][push](local$$36100);
            }
          }
        }
      } else {
        if (local$$37126 instanceof THREE[Line]) {
          if (local$$37131 instanceof THREE[BufferGeometry]) {
            local$$37157 = local$$37131[attributes];
            if (local$$37157[position] !== undefined) {
              local$$37178 = local$$37157[position][array];
              /** @type {number} */
              local$$37181 = 0;
              local$$37186 = local$$37178[length];
              for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
                local$$36868[pushVertex](local$$37178[local$$37181], local$$37178[local$$37181 + 1], local$$37178[local$$37181 + 2]);
              }
              if (local$$37131[index] !== null) {
                local$$37300 = local$$37131[index][array];
                /** @type {number} */
                local$$37181 = 0;
                local$$37186 = local$$37300[length];
                for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 2) {
                  local$$36868[pushLine](local$$37300[local$$37181], local$$37300[local$$37181 + 1]);
                }
              } else {
                /** @type {number} */
                var local$$38026 = local$$37126 instanceof THREE[LineSegments] ? 2 : 1;
                /** @type {number} */
                local$$37181 = 0;
                /** @type {number} */
                local$$37186 = local$$37178[length] / 3 - 1;
                for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + local$$38026) {
                  local$$36868[pushLine](local$$37181, local$$37181 + 1);
                }
              }
            }
          } else {
            if (local$$37131 instanceof THREE[Geometry]) {
              local$$36174[multiplyMatrices](local$$36168, local$$36170);
              local$$37421 = local$$37126[geometry][vertices];
              if (local$$37421[length] === 0) {
                continue;
              }
              local$$37627 = local$$35758();
              local$$37627[positionScreen][copy](local$$37421[0])[applyMatrix4](local$$36174);
              /** @type {number} */
              local$$38026 = local$$37126 instanceof THREE[LineSegments] ? 2 : 1;
              /** @type {number} */
              local$$37461 = 1;
              local$$37466 = local$$37421[length];
              for (; local$$37461 < local$$37466; local$$37461++) {
                local$$37627 = local$$35758();
                local$$37627[positionScreen][copy](local$$37421[local$$37461])[applyMatrix4](local$$36174);
                if ((local$$37461 + 1) % local$$38026 > 0) {
                  continue;
                }
                local$$37633 = local$$35767[local$$35760 - 2];
                local$$36188[copy](local$$37627[positionScreen]);
                local$$36192[copy](local$$37633[positionScreen]);
                if (local$$35932(local$$36188, local$$36192) === true) {
                  local$$36188[multiplyScalar](1 / local$$36188[w]);
                  local$$36192[multiplyScalar](1 / local$$36192[w]);
                  local$$36107 = local$$35814();
                  local$$36107[id] = local$$37126[id];
                  local$$36107[v1][positionScreen][copy](local$$36188);
                  local$$36107[v2][positionScreen][copy](local$$36192);
                  local$$36107[z] = Math[max](local$$36188[z], local$$36192[z]);
                  local$$36107[renderOrder] = local$$37126[renderOrder];
                  local$$36107[material] = local$$37126[material];
                  if (local$$37126[material][vertexColors] === THREE[VertexColors]) {
                    local$$36107[vertexColors][0][copy](local$$37126[geometry][colors][local$$37461]);
                    local$$36107[vertexColors][1][copy](local$$37126[geometry][colors][local$$37461 - 1]);
                  }
                  local$$36125[elements][push](local$$36107);
                }
              }
            }
          }
        } else {
          if (local$$37126 instanceof THREE[Sprite]) {
            local$$36133[set](local$$36170[elements][12], local$$36170[elements][13], local$$36170[elements][14], 1);
            local$$36133[applyMatrix4](local$$36168);
            /** @type {number} */
            var local$$38355 = 1 / local$$36133[w];
            local$$36133[z] *= local$$38355;
            if (local$$36133[z] >= -1 && local$$36133[z] <= 1) {
              local$$36114 = local$$35842();
              local$$36114[id] = local$$37126[id];
              /** @type {number} */
              local$$36114[x] = local$$36133[x] * local$$38355;
              /** @type {number} */
              local$$36114[y] = local$$36133[y] * local$$38355;
              local$$36114[z] = local$$36133[z];
              local$$36114[renderOrder] = local$$37126[renderOrder];
              local$$36114[object] = local$$37126;
              local$$36114[rotation] = local$$37126[rotation];
              /** @type {number} */
              local$$36114[scale][x] = local$$37126[scale][x] * Math[abs](local$$36114[x] - (local$$36133[x] + local$$36874[projectionMatrix][elements][0]) / (local$$36133[w] + local$$36874[projectionMatrix][elements][12]));
              /** @type {number} */
              local$$36114[scale][y] = local$$37126[scale][y] * Math[abs](local$$36114[y] - (local$$36133[y] + local$$36874[projectionMatrix][elements][5]) / (local$$36133[w] + local$$36874[projectionMatrix][elements][13]));
              local$$36114[material] = local$$37126[material];
              local$$36125[elements][push](local$$36114);
            }
          }
        }
      }
    }
    if (local$$36876 === true) {
      local$$36125[elements][sort](local$$35870);
    }
    return local$$36125;
  };
};
(function() {
  use strict;
  /**
   * @param {?} local$$38582
   * @return {undefined}
   */
  var local$$38693 = function(local$$38582) {
    THREE[MeshBasicMaterial][call](this);
    /** @type {boolean} */
    this[depthTest] = false;
    /** @type {boolean} */
    this[depthWrite] = false;
    this[side] = THREE[FrontSide];
    /** @type {boolean} */
    this[transparent] = true;
    this[setValues](local$$38582);
    this[oldColor] = this[color][clone]();
    this[oldOpacity] = this[opacity];
    /**
     * @param {?} local$$38646
     * @return {undefined}
     */
    this[highlight] = function(local$$38646) {
      if (local$$38646) {
        this[color][setRGB](1, 1, 0);
        /** @type {number} */
        this[opacity] = 1;
      } else {
        this[color][copy](this[oldColor]);
        this[opacity] = this[oldOpacity];
      }
    };
  };
  local$$38693[prototype] = Object[create](THREE[MeshBasicMaterial][prototype]);
  /** @type {function(?): undefined} */
  local$$38693[prototype][constructor] = local$$38693;
  /**
   * @param {?} local$$38718
   * @return {undefined}
   */
  var local$$38827 = function(local$$38718) {
    THREE[LineBasicMaterial][call](this);
    /** @type {boolean} */
    this[depthTest] = false;
    /** @type {boolean} */
    this[depthWrite] = false;
    /** @type {boolean} */
    this[transparent] = true;
    /** @type {number} */
    this[linewidth] = 1;
    this[setValues](local$$38718);
    this[oldColor] = this[color][clone]();
    this[oldOpacity] = this[opacity];
    /**
     * @param {?} local$$38780
     * @return {undefined}
     */
    this[highlight] = function(local$$38780) {
      if (local$$38780) {
        this[color][setRGB](1, 1, 0);
        /** @type {number} */
        this[opacity] = 1;
      } else {
        this[color][copy](this[oldColor]);
        this[opacity] = this[oldOpacity];
      }
    };
  };
  local$$38827[prototype] = Object[create](THREE[LineBasicMaterial][prototype]);
  /** @type {function(?): undefined} */
  local$$38827[prototype][constructor] = local$$38827;
  var local$$38856 = new local$$38693({
    visible : false,
    transparent : false
  });
  /**
   * @return {undefined}
   */
  THREE[TransformGizmo] = function() {
    var local$$38862 = this;
    /**
     * @return {undefined}
     */
    this[init] = function() {
      THREE[Object3D][call](this);
      this[handles] = new THREE.Object3D;
      this[pickers] = new THREE.Object3D;
      this[planes] = new THREE.Object3D;
      this[add](this[handles]);
      this[add](this[pickers]);
      this[add](this[planes]);
      var local$$38927 = new THREE.PlaneBufferGeometry(50, 50, 2, 2);
      var local$$38936 = new THREE.MeshBasicMaterial({
        visible : false,
        side : THREE[DoubleSide]
      });
      var local$$38947 = {
        "XY" : new THREE.Mesh(local$$38927, local$$38936),
        "YZ" : new THREE.Mesh(local$$38927, local$$38936),
        "XZ" : new THREE.Mesh(local$$38927, local$$38936),
        "XYZE" : new THREE.Mesh(local$$38927, local$$38936)
      };
      this[activePlane] = local$$38947[XYZE];
      local$$38947[YZ][rotation][set](0, Math[PI] / 2, 0);
      local$$38947[XZ][rotation][set](-Math[PI] / 2, 0, 0);
      var local$$38994;
      for (local$$38994 in local$$38947) {
        local$$38947[local$$38994][name] = local$$38994;
        this[planes][add](local$$38947[local$$38994]);
        this[planes][local$$38994] = local$$38947[local$$38994];
      }
      /**
       * @param {!Object} local$$39021
       * @param {?} local$$39022
       * @return {undefined}
       */
      var local$$39103 = function(local$$39021, local$$39022) {
        var local$$39024;
        for (local$$39024 in local$$39021) {
          local$$38994 = local$$39021[local$$39024][length];
          for (; local$$38994--;) {
            var local$$39039 = local$$39021[local$$39024][local$$38994][0];
            var local$$39045 = local$$39021[local$$39024][local$$38994][1];
            var local$$39051 = local$$39021[local$$39024][local$$38994][2];
            /** @type {string} */
            local$$39039[name] = local$$39024;
            if (local$$39045) {
              local$$39039[position][set](local$$39045[0], local$$39045[1], local$$39045[2]);
            }
            if (local$$39051) {
              local$$39039[rotation][set](local$$39051[0], local$$39051[1], local$$39051[2]);
            }
            local$$39022[add](local$$39039);
          }
        }
      };
      local$$39103(this[handleGizmos], this[handles]);
      local$$39103(this[pickerGizmos], this[pickers]);
      this[traverse](function(local$$39124) {
        if (local$$39124 instanceof THREE[Mesh]) {
          local$$39124[updateMatrix]();
          var local$$39142 = local$$39124[geometry][clone]();
          local$$39142[applyMatrix](local$$39124[matrix]);
          local$$39124[geometry] = local$$39142;
          local$$39124[position][set](0, 0, 0);
          local$$39124[rotation][set](0, 0, 0);
          local$$39124[scale][set](1, 1, 1);
        }
      });
    };
    /**
     * @param {?} local$$39203
     * @return {undefined}
     */
    this[highlight] = function(local$$39203) {
      this[traverse](function(local$$39208) {
        if (local$$39208[material] && local$$39208[material][highlight]) {
          if (local$$39208[name] === local$$39203) {
            local$$39208[material][highlight](true);
          } else {
            local$$39208[material][highlight](false);
          }
        }
      });
    };
  };
  THREE[TransformGizmo][prototype] = Object[create](THREE[Object3D][prototype]);
  THREE[TransformGizmo][prototype][constructor] = THREE[TransformGizmo];
  /**
   * @param {?} local$$39300
   * @param {?} local$$39301
   * @return {undefined}
   */
  THREE[TransformGizmo][prototype][update] = function(local$$39300, local$$39301) {
    var local$$39308 = new THREE.Vector3(0, 0, 0);
    var local$$39315 = new THREE.Vector3(0, 1, 0);
    var local$$39319 = new THREE.Matrix4;
    this[traverse](function(local$$39324) {
      if (local$$39324[name][search](E) !== -1) {
        local$$39324[quaternion][setFromRotationMatrix](local$$39319[lookAt](local$$39301, local$$39308, local$$39315));
      } else {
        if (local$$39324[name][search](X) !== -1 || local$$39324[name][search](Y) !== -1 || local$$39324[name][search](Z) !== -1) {
          local$$39324[quaternion][setFromEuler](local$$39300);
        }
      }
    });
  };
  /**
   * @return {undefined}
   */
  THREE[TransformGizmoTranslate] = function() {
    THREE[TransformGizmo][call](this);
    var local$$39419 = new THREE.Geometry;
    var local$$39432 = new THREE.Mesh(new THREE.CylinderGeometry(0, .05, .2, 12, 1, false));
    /** @type {number} */
    local$$39432[position][y] = .5;
    local$$39432[updateMatrix]();
    local$$39419[merge](local$$39432[geometry], local$$39432[matrix]);
    var local$$39461 = new THREE.BufferGeometry;
    local$$39461[addAttribute](position, new THREE.Float32Attribute([0, 0, 0, 1, 0, 0], 3));
    var local$$39483 = new THREE.BufferGeometry;
    local$$39483[addAttribute](position, new THREE.Float32Attribute([0, 0, 0, 0, 1, 0], 3));
    var local$$39504 = new THREE.BufferGeometry;
    local$$39504[addAttribute](position, new THREE.Float32Attribute([0, 0, 0, 0, 0, 1], 3));
    this[handleGizmos] = {
      X : [[new THREE.Mesh(local$$39419, new local$$38693({
        color : 16711680
      })), [.5, 0, 0], [0, 0, -Math[PI] / 2]], [new THREE.Line(local$$39461, new local$$38827({
        color : 16711680
      }))]],
      Y : [[new THREE.Mesh(local$$39419, new local$$38693({
        color : 65280
      })), [0, .5, 0]], [new THREE.Line(local$$39483, new local$$38827({
        color : 65280
      }))]],
      Z : [[new THREE.Mesh(local$$39419, new local$$38693({
        color : 255
      })), [0, 0, .5], [Math[PI] / 2, 0, 0]], [new THREE.Line(local$$39504, new local$$38827({
        color : 255
      }))]],
      XYZ : [[new THREE.Mesh(new THREE.OctahedronGeometry(.1, 0), new local$$38693({
        color : 16777215,
        opacity : .25
      })), [0, 0, 0], [0, 0, 0]]],
      XY : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.29, .29), new local$$38693({
        color : 16776960,
        opacity : .25
      })), [.15, .15, 0]]],
      YZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.29, .29), new local$$38693({
        color : 65535,
        opacity : .25
      })), [0, .15, .15], [0, Math[PI] / 2, 0]]],
      XZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.29, .29), new local$$38693({
        color : 16711935,
        opacity : .25
      })), [.15, 0, .15], [-Math[PI] / 2, 0, 0]]]
    };
    this[pickerGizmos] = {
      X : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [.6, 0, 0], [0, 0, -Math[PI] / 2]]],
      Y : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, .6, 0]]],
      Z : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, 0, .6], [Math[PI] / 2, 0, 0]]],
      XYZ : [[new THREE.Mesh(new THREE.OctahedronGeometry(.2, 0), local$$38856)]],
      XY : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.4, .4), local$$38856), [.2, .2, 0]]],
      YZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.4, .4), local$$38856), [0, .2, .2], [0, Math[PI] / 2, 0]]],
      XZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.4, .4), local$$38856), [.2, 0, .2], [-Math[PI] / 2, 0, 0]]]
    };
    /**
     * @param {?} local$$39818
     * @param {?} local$$39819
     * @return {undefined}
     */
    this[setActivePlane] = function(local$$39818, local$$39819) {
      var local$$39823 = new THREE.Matrix4;
      local$$39819[applyMatrix4](local$$39823[getInverse](local$$39823[extractRotation](this[planes][XY][matrixWorld])));
      if (local$$39818 === X) {
        this[activePlane] = this[planes][XY];
        if (Math[abs](local$$39819[y]) > Math[abs](local$$39819[z])) {
          this[activePlane] = this[planes][XZ];
        }
      }
      if (local$$39818 === Y) {
        this[activePlane] = this[planes][XY];
        if (Math[abs](local$$39819[x]) > Math[abs](local$$39819[z])) {
          this[activePlane] = this[planes][YZ];
        }
      }
      if (local$$39818 === Z) {
        this[activePlane] = this[planes][XZ];
        if (Math[abs](local$$39819[x]) > Math[abs](local$$39819[y])) {
          this[activePlane] = this[planes][YZ];
        }
      }
      if (local$$39818 === XYZ) {
        this[activePlane] = this[planes][XYZE];
      }
      if (local$$39818 === XY) {
        this[activePlane] = this[planes][XY];
      }
      if (local$$39818 === YZ) {
        this[activePlane] = this[planes][YZ];
      }
      if (local$$39818 === XZ) {
        this[activePlane] = this[planes][XZ];
      }
    };
    this[init]();
  };
  THREE[TransformGizmoTranslate][prototype] = Object[create](THREE[TransformGizmo][prototype]);
  THREE[TransformGizmoTranslate][prototype][constructor] = THREE[TransformGizmoTranslate];
  /**
   * @return {undefined}
   */
  THREE[TransformGizmoRotate] = function() {
    THREE[TransformGizmo][call](this);
    /**
     * @param {?} local$$40106
     * @param {?} local$$40107
     * @param {number} local$$40108
     * @return {?}
     */
    var local$$40246 = function(local$$40106, local$$40107, local$$40108) {
      var local$$40112 = new THREE.BufferGeometry;
      /** @type {!Array} */
      var local$$40115 = [];
      local$$40108 = local$$40108 ? local$$40108 : 1;
      /** @type {number} */
      var local$$40122 = 0;
      for (; local$$40122 <= 64 * local$$40108; ++local$$40122) {
        if (local$$40107 === x) {
          local$$40115[push](0, Math[cos](local$$40122 / 32 * Math[PI]) * local$$40106, Math[sin](local$$40122 / 32 * Math[PI]) * local$$40106);
        }
        if (local$$40107 === y) {
          local$$40115[push](Math[cos](local$$40122 / 32 * Math[PI]) * local$$40106, 0, Math[sin](local$$40122 / 32 * Math[PI]) * local$$40106);
        }
        if (local$$40107 === z) {
          local$$40115[push](Math[sin](local$$40122 / 32 * Math[PI]) * local$$40106, Math[cos](local$$40122 / 32 * Math[PI]) * local$$40106, 0);
        }
      }
      local$$40112[addAttribute](position, new THREE.Float32Attribute(local$$40115, 3));
      return local$$40112;
    };
    this[handleGizmos] = {
      X : [[new THREE.Line(new local$$40246(1, x, .5), new local$$38827({
        color : 16711680
      }))]],
      Y : [[new THREE.Line(new local$$40246(1, y, .5), new local$$38827({
        color : 65280
      }))]],
      Z : [[new THREE.Line(new local$$40246(1, z, .5), new local$$38827({
        color : 255
      }))]],
      E : [[new THREE.Line(new local$$40246(1.25, z, 1), new local$$38827({
        color : 13421568
      }))]],
      XYZE : [[new THREE.Line(new local$$40246(1, z, 1), new local$$38827({
        color : 7895160
      }))]]
    };
    this[pickerGizmos] = {
      X : [[new THREE.Mesh(new THREE.TorusGeometry(1, .12, 4, 12, Math.PI), local$$38856), [0, 0, 0], [0, -Math[PI] / 2, -Math[PI] / 2]]],
      Y : [[new THREE.Mesh(new THREE.TorusGeometry(1, .12, 4, 12, Math.PI), local$$38856), [0, 0, 0], [Math[PI] / 2, 0, 0]]],
      Z : [[new THREE.Mesh(new THREE.TorusGeometry(1, .12, 4, 12, Math.PI), local$$38856), [0, 0, 0], [0, 0, -Math[PI] / 2]]],
      E : [[new THREE.Mesh(new THREE.TorusGeometry(1.25, .12, 2, 24), local$$38856)]],
      XYZE : [[new THREE.Mesh(new THREE.Geometry)]]
    };
    /**
     * @param {?} local$$40416
     * @return {undefined}
     */
    this[setActivePlane] = function(local$$40416) {
      if (local$$40416 === E) {
        this[activePlane] = this[planes][XYZE];
      }
      if (local$$40416 === X) {
        this[activePlane] = this[planes][YZ];
      }
      if (local$$40416 === Y) {
        this[activePlane] = this[planes][XZ];
      }
      if (local$$40416 === Z) {
        this[activePlane] = this[planes][XY];
      }
    };
    /**
     * @param {?} local$$40492
     * @param {?} local$$40493
     * @return {undefined}
     */
    this[update] = function(local$$40492, local$$40493) {
      THREE[TransformGizmo][prototype][update][apply](this, arguments);
      var local$$40516 = {
        handles : this[handles],
        pickers : this[pickers]
      };
      var local$$40520 = new THREE.Matrix4;
      var local$$40528 = new THREE.Euler(0, 0, 1);
      var local$$40532 = new THREE.Quaternion;
      var local$$40539 = new THREE.Vector3(1, 0, 0);
      var local$$40546 = new THREE.Vector3(0, 1, 0);
      var local$$40553 = new THREE.Vector3(0, 0, 1);
      var local$$40557 = new THREE.Quaternion;
      var local$$40561 = new THREE.Quaternion;
      var local$$40565 = new THREE.Quaternion;
      var local$$40571 = local$$40493[clone]();
      local$$40528[copy](this[planes][XY][rotation]);
      local$$40532[setFromEuler](local$$40528);
      local$$40520[makeRotationFromQuaternion](local$$40532)[getInverse](local$$40520);
      local$$40571[applyMatrix4](local$$40520);
      this[traverse](function(local$$40609) {
        local$$40532[setFromEuler](local$$40528);
        if (local$$40609[name] === X) {
          local$$40557[setFromAxisAngle](local$$40539, Math[atan2](-local$$40571[y], local$$40571[z]));
          local$$40532[multiplyQuaternions](local$$40532, local$$40557);
          local$$40609[quaternion][copy](local$$40532);
        }
        if (local$$40609[name] === Y) {
          local$$40561[setFromAxisAngle](local$$40546, Math[atan2](local$$40571[x], local$$40571[z]));
          local$$40532[multiplyQuaternions](local$$40532, local$$40561);
          local$$40609[quaternion][copy](local$$40532);
        }
        if (local$$40609[name] === Z) {
          local$$40565[setFromAxisAngle](local$$40553, Math[atan2](local$$40571[y], local$$40571[x]));
          local$$40532[multiplyQuaternions](local$$40532, local$$40565);
          local$$40609[quaternion][copy](local$$40532);
        }
      });
    };
    this[init]();
  };
  THREE[TransformGizmoRotate][prototype] = Object[create](THREE[TransformGizmo][prototype]);
  THREE[TransformGizmoRotate][prototype][constructor] = THREE[TransformGizmoRotate];
  /**
   * @return {undefined}
   */
  THREE[TransformGizmoScale] = function() {
    THREE[TransformGizmo][call](this);
    var local$$40790 = new THREE.Geometry;
    var local$$40800 = new THREE.Mesh(new THREE.BoxGeometry(.125, .125, .125));
    /** @type {number} */
    local$$40800[position][y] = .5;
    local$$40800[updateMatrix]();
    local$$40790[merge](local$$40800[geometry], local$$40800[matrix]);
    var local$$40829 = new THREE.BufferGeometry;
    local$$40829[addAttribute](position, new THREE.Float32Attribute([0, 0, 0, 1, 0, 0], 3));
    var local$$40850 = new THREE.BufferGeometry;
    local$$40850[addAttribute](position, new THREE.Float32Attribute([0, 0, 0, 0, 1, 0], 3));
    var local$$40871 = new THREE.BufferGeometry;
    local$$40871[addAttribute](position, new THREE.Float32Attribute([0, 0, 0, 0, 0, 1], 3));
    this[handleGizmos] = {
      X : [[new THREE.Mesh(local$$40790, new local$$38693({
        color : 16711680
      })), [.5, 0, 0], [0, 0, -Math[PI] / 2]], [new THREE.Line(local$$40829, new local$$38827({
        color : 16711680
      }))]],
      Y : [[new THREE.Mesh(local$$40790, new local$$38693({
        color : 65280
      })), [0, .5, 0]], [new THREE.Line(local$$40850, new local$$38827({
        color : 65280
      }))]],
      Z : [[new THREE.Mesh(local$$40790, new local$$38693({
        color : 255
      })), [0, 0, .5], [Math[PI] / 2, 0, 0]], [new THREE.Line(local$$40871, new local$$38827({
        color : 255
      }))]],
      XYZ : [[new THREE.Mesh(new THREE.BoxGeometry(.125, .125, .125), new local$$38693({
        color : 16777215,
        opacity : .25
      }))]]
    };
    this[pickerGizmos] = {
      X : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [.6, 0, 0], [0, 0, -Math[PI] / 2]]],
      Y : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, .6, 0]]],
      Z : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, 0, .6], [Math[PI] / 2, 0, 0]]],
      XYZ : [[new THREE.Mesh(new THREE.BoxGeometry(.4, .4, .4), local$$38856)]]
    };
    /**
     * @param {?} local$$41060
     * @param {?} local$$41061
     * @return {undefined}
     */
    this[setActivePlane] = function(local$$41060, local$$41061) {
      var local$$41065 = new THREE.Matrix4;
      local$$41061[applyMatrix4](local$$41065[getInverse](local$$41065[extractRotation](this[planes][XY][matrixWorld])));
      if (local$$41060 === X) {
        this[activePlane] = this[planes][XY];
        if (Math[abs](local$$41061[y]) > Math[abs](local$$41061[z])) {
          this[activePlane] = this[planes][XZ];
        }
      }
      if (local$$41060 === Y) {
        this[activePlane] = this[planes][XY];
        if (Math[abs](local$$41061[x]) > Math[abs](local$$41061[z])) {
          this[activePlane] = this[planes][YZ];
        }
      }
      if (local$$41060 === Z) {
        this[activePlane] = this[planes][XZ];
        if (Math[abs](local$$41061[x]) > Math[abs](local$$41061[y])) {
          this[activePlane] = this[planes][YZ];
        }
      }
      if (local$$41060 === XYZ) {
        this[activePlane] = this[planes][XYZE];
      }
    };
    this[init]();
  };
  THREE[TransformGizmoScale][prototype] = Object[create](THREE[TransformGizmo][prototype]);
  THREE[TransformGizmoScale][prototype][constructor] = THREE[TransformGizmoScale];
  /**
   * @param {?} local$$41288
   * @param {!Object} local$$41289
   * @return {undefined}
   */
  THREE[TransformControls] = function(local$$41288, local$$41289) {
    /**
     * @param {?} local$$41292
     * @return {undefined}
     */
    function local$$41291(local$$41292) {
      if (local$$41294[object] === undefined || local$$41299 === true || local$$41292[button] !== undefined && local$$41292[button] !== 0) {
        return;
      }
      var local$$41327 = local$$41292[changedTouches] ? local$$41292[changedTouches][0] : local$$41292;
      var local$$41340 = local$$41329(local$$41327, local$$41330[local$$41331][pickers][children]);
      /** @type {null} */
      var local$$41343 = null;
      if (local$$41340) {
        local$$41343 = local$$41340[object][name];
        local$$41292[preventDefault]();
      }
      if (local$$41294[axis] !== local$$41343) {
        local$$41294[axis] = local$$41343;
        local$$41294[update]();
        local$$41294[dispatchEvent](local$$41378);
      }
    }
    /**
     * @param {?} local$$41386
     * @return {undefined}
     */
    function local$$41385(local$$41386) {
      if (local$$41294[object] === undefined || local$$41299 === true || local$$41386[button] !== undefined && local$$41386[button] !== 0) {
        return;
      }
      var local$$41419 = local$$41386[changedTouches] ? local$$41386[changedTouches][0] : local$$41386;
      if (local$$41419[button] === 0 || local$$41419[button] === undefined) {
        var local$$41439 = local$$41329(local$$41419, local$$41330[local$$41331][pickers][children]);
        if (local$$41439) {
          local$$41386[preventDefault]();
          local$$41386[stopPropagation]();
          local$$41294[dispatchEvent](local$$41454);
          local$$41294[axis] = local$$41439[object][name];
          local$$41294[update]();
          local$$41473[copy](local$$41477)[sub](local$$41482)[normalize]();
          local$$41330[local$$41331][setActivePlane](local$$41294[axis], local$$41473);
          var local$$41504 = local$$41329(local$$41419, [local$$41330[local$$41331][activePlane]]);
          if (local$$41504) {
            local$$41506[copy](local$$41294[object][position]);
            local$$41518[copy](local$$41294[object][scale]);
            local$$41530[extractRotation](local$$41294[object][matrix]);
            local$$41542[extractRotation](local$$41294[object][matrixWorld]);
            local$$41554[extractRotation](local$$41294[object][parent][matrixWorld]);
            local$$41569[setFromMatrixScale](local$$41573[getInverse](local$$41294[object][parent][matrixWorld]));
            local$$41589[copy](local$$41504[point]);
          }
        }
      }
      /** @type {boolean} */
      local$$41299 = true;
    }
    /**
     * @param {?} local$$41611
     * @return {undefined}
     */
    function local$$41610(local$$41611) {
      if (local$$41294[object] === undefined || local$$41294[axis] === null || local$$41299 === false || local$$41611[button] !== undefined && local$$41611[button] !== 0) {
        return;
      }
      var local$$41650 = local$$41611[changedTouches] ? local$$41611[changedTouches][0] : local$$41611;
      var local$$41658 = local$$41329(local$$41650, [local$$41330[local$$41331][activePlane]]);
      if (local$$41658 === false) {
        return;
      }
      local$$41611[preventDefault]();
      local$$41611[stopPropagation]();
      local$$41676[copy](local$$41658[point]);
      if (local$$41331 === translate) {
        local$$41676[sub](local$$41589);
        local$$41676[multiply](local$$41569);
        if (local$$41294[space] === local) {
          local$$41676[applyMatrix4](local$$41573[getInverse](local$$41542));
          if (local$$41294[axis][search](X) === -1) {
            /** @type {number} */
            local$$41676[x] = 0;
          }
          if (local$$41294[axis][search](Y) === -1) {
            /** @type {number} */
            local$$41676[y] = 0;
          }
          if (local$$41294[axis][search](Z) === -1) {
            /** @type {number} */
            local$$41676[z] = 0;
          }
          local$$41676[applyMatrix4](local$$41530);
          local$$41294[object][position][copy](local$$41506);
          local$$41294[object][position][add](local$$41676);
        }
        if (local$$41294[space] === world || local$$41294[axis][search](XYZ) !== -1) {
          if (local$$41294[axis][search](X) === -1) {
            /** @type {number} */
            local$$41676[x] = 0;
          }
          if (local$$41294[axis][search](Y) === -1) {
            /** @type {number} */
            local$$41676[y] = 0;
          }
          if (local$$41294[axis][search](Z) === -1) {
            /** @type {number} */
            local$$41676[z] = 0;
          }
          local$$41676[applyMatrix4](local$$41573[getInverse](local$$41554));
          local$$41294[object][position][copy](local$$41506);
          local$$41294[object][position][add](local$$41676);
        }
        if (local$$41294[translationSnap] !== null) {
          if (local$$41294[space] === local) {
            local$$41294[object][position][applyMatrix4](local$$41573[getInverse](local$$41542));
          }
          if (local$$41294[axis][search](X) !== -1) {
            /** @type {number} */
            local$$41294[object][position][x] = Math[round](local$$41294[object][position][x] / local$$41294[translationSnap]) * local$$41294[translationSnap];
          }
          if (local$$41294[axis][search](Y) !== -1) {
            /** @type {number} */
            local$$41294[object][position][y] = Math[round](local$$41294[object][position][y] / local$$41294[translationSnap]) * local$$41294[translationSnap];
          }
          if (local$$41294[axis][search](Z) !== -1) {
            /** @type {number} */
            local$$41294[object][position][z] = Math[round](local$$41294[object][position][z] / local$$41294[translationSnap]) * local$$41294[translationSnap];
          }
          if (local$$41294[space] === local) {
            local$$41294[object][position][applyMatrix4](local$$41542);
          }
        }
      } else {
        if (local$$41331 === scale) {
          local$$41676[sub](local$$41589);
          local$$41676[multiply](local$$41569);
          if (local$$41294[space] === local) {
            if (local$$41294[axis] === XYZ) {
              /** @type {number} */
              local$$42129 = 1 + local$$41676[y] / 50;
              /** @type {number} */
              local$$41294[object][scale][x] = local$$41518[x] * local$$42129;
              /** @type {number} */
              local$$41294[object][scale][y] = local$$41518[y] * local$$42129;
              /** @type {number} */
              local$$41294[object][scale][z] = local$$41518[z] * local$$42129;
            } else {
              local$$41676[applyMatrix4](local$$41573[getInverse](local$$41542));
              if (local$$41294[axis] === X) {
                /** @type {number} */
                local$$41294[object][scale][x] = local$$41518[x] * (1 + local$$41676[x] / 50);
              }
              if (local$$41294[axis] === Y) {
                /** @type {number} */
                local$$41294[object][scale][y] = local$$41518[y] * (1 + local$$41676[y] / 50);
              }
              if (local$$41294[axis] === Z) {
                /** @type {number} */
                local$$41294[object][scale][z] = local$$41518[z] * (1 + local$$41676[z] / 50);
              }
            }
          }
        } else {
          if (local$$41331 === rotate) {
            local$$41676[sub](local$$41482);
            local$$41676[multiply](local$$41569);
            local$$42304[copy](local$$41589)[sub](local$$41482);
            local$$42304[multiply](local$$41569);
            if (local$$41294[axis] === E) {
              local$$41676[applyMatrix4](local$$41573[getInverse](local$$42331));
              local$$42304[applyMatrix4](local$$41573[getInverse](local$$42331));
              local$$42344[set](Math[atan2](local$$41676[z], local$$41676[y]), Math[atan2](local$$41676[x], local$$41676[z]), Math[atan2](local$$41676[y], local$$41676[x]));
              local$$42380[set](Math[atan2](local$$42304[z], local$$42304[y]), Math[atan2](local$$42304[x], local$$42304[z]), Math[atan2](local$$42304[y], local$$42304[x]));
              local$$42416[setFromRotationMatrix](local$$41573[getInverse](local$$41554));
              local$$42426[setFromAxisAngle](local$$41473, local$$42344[z] - local$$42380[z]);
              local$$42439[setFromRotationMatrix](local$$41542);
              local$$42416[multiplyQuaternions](local$$42416, local$$42426);
              local$$42416[multiplyQuaternions](local$$42416, local$$42439);
              local$$41294[object][quaternion][copy](local$$42416);
            } else {
              if (local$$41294[axis] === XYZE) {
                local$$42426[setFromEuler](local$$41676[clone]()[cross](local$$42304)[normalize]());
                local$$42416[setFromRotationMatrix](local$$41573[getInverse](local$$41554));
                local$$42499[setFromAxisAngle](local$$42426, -local$$41676[clone]()[angleTo](local$$42304));
                local$$42439[setFromRotationMatrix](local$$41542);
                local$$42416[multiplyQuaternions](local$$42416, local$$42499);
                local$$42416[multiplyQuaternions](local$$42416, local$$42439);
                local$$41294[object][quaternion][copy](local$$42416);
              } else {
                if (local$$41294[space] === local) {
                  local$$41676[applyMatrix4](local$$41573[getInverse](local$$41542));
                  local$$42304[applyMatrix4](local$$41573[getInverse](local$$41542));
                  local$$42344[set](Math[atan2](local$$41676[z], local$$41676[y]), Math[atan2](local$$41676[x], local$$41676[z]), Math[atan2](local$$41676[y], local$$41676[x]));
                  local$$42380[set](Math[atan2](local$$42304[z], local$$42304[y]), Math[atan2](local$$42304[x], local$$42304[z]), Math[atan2](local$$42304[y], local$$42304[x]));
                  local$$42439[setFromRotationMatrix](local$$41530);
                  if (local$$41294[rotationSnap] !== null) {
                    local$$42499[setFromAxisAngle](local$$42648, Math[round]((local$$42344[x] - local$$42380[x]) / local$$41294[rotationSnap]) * local$$41294[rotationSnap]);
                    local$$42670[setFromAxisAngle](local$$42674, Math[round]((local$$42344[y] - local$$42380[y]) / local$$41294[rotationSnap]) * local$$41294[rotationSnap]);
                    local$$42696[setFromAxisAngle](local$$42700, Math[round]((local$$42344[z] - local$$42380[z]) / local$$41294[rotationSnap]) * local$$41294[rotationSnap]);
                  } else {
                    local$$42499[setFromAxisAngle](local$$42648, local$$42344[x] - local$$42380[x]);
                    local$$42670[setFromAxisAngle](local$$42674, local$$42344[y] - local$$42380[y]);
                    local$$42696[setFromAxisAngle](local$$42700, local$$42344[z] - local$$42380[z]);
                  }
                  if (local$$41294[axis] === X) {
                    local$$42439[multiplyQuaternions](local$$42439, local$$42499);
                  }
                  if (local$$41294[axis] === Y) {
                    local$$42439[multiplyQuaternions](local$$42439, local$$42670);
                  }
                  if (local$$41294[axis] === Z) {
                    local$$42439[multiplyQuaternions](local$$42439, local$$42696);
                  }
                  local$$41294[object][quaternion][copy](local$$42439);
                } else {
                  if (local$$41294[space] === world) {
                    local$$42344[set](Math[atan2](local$$41676[z], local$$41676[y]), Math[atan2](local$$41676[x], local$$41676[z]), Math[atan2](local$$41676[y], local$$41676[x]));
                    local$$42380[set](Math[atan2](local$$42304[z], local$$42304[y]), Math[atan2](local$$42304[x], local$$42304[z]), Math[atan2](local$$42304[y], local$$42304[x]));
                    local$$42416[setFromRotationMatrix](local$$41573[getInverse](local$$41554));
                    if (local$$41294[rotationSnap] !== null) {
                      local$$42499[setFromAxisAngle](local$$42648, Math[round]((local$$42344[x] - local$$42380[x]) / local$$41294[rotationSnap]) * local$$41294[rotationSnap]);
                      local$$42670[setFromAxisAngle](local$$42674, Math[round]((local$$42344[y] - local$$42380[y]) / local$$41294[rotationSnap]) * local$$41294[rotationSnap]);
                      local$$42696[setFromAxisAngle](local$$42700, Math[round]((local$$42344[z] - local$$42380[z]) / local$$41294[rotationSnap]) * local$$41294[rotationSnap]);
                    } else {
                      local$$42499[setFromAxisAngle](local$$42648, local$$42344[x] - local$$42380[x]);
                      local$$42670[setFromAxisAngle](local$$42674, local$$42344[y] - local$$42380[y]);
                      local$$42696[setFromAxisAngle](local$$42700, local$$42344[z] - local$$42380[z]);
                    }
                    local$$42439[setFromRotationMatrix](local$$41542);
                    if (local$$41294[axis] === X) {
                      local$$42416[multiplyQuaternions](local$$42416, local$$42499);
                    }
                    if (local$$41294[axis] === Y) {
                      local$$42416[multiplyQuaternions](local$$42416, local$$42670);
                    }
                    if (local$$41294[axis] === Z) {
                      local$$42416[multiplyQuaternions](local$$42416, local$$42696);
                    }
                    local$$42416[multiplyQuaternions](local$$42416, local$$42439);
                    local$$41294[object][quaternion][copy](local$$42416);
                  }
                }
              }
            }
          }
        }
      }
      local$$41294[update]();
      local$$41294[dispatchEvent](local$$41378);
      local$$41294[dispatchEvent](local$$43109);
    }
    /**
     * @param {?} local$$43115
     * @return {undefined}
     */
    function local$$43114(local$$43115) {
      if (local$$43115[button] !== undefined && local$$43115[button] !== 0) {
        return;
      }
      if (local$$41299 && local$$41294[axis] !== null) {
        local$$43137[mode] = local$$41331;
        local$$41294[dispatchEvent](local$$43137);
      }
      /** @type {boolean} */
      local$$41299 = false;
      local$$41291(local$$43115);
    }
    /**
     * @param {?} local$$43158
     * @param {!Array} local$$43159
     * @return {?}
     */
    function local$$41329(local$$43158, local$$43159) {
      var local$$43165 = local$$41289[getBoundingClientRect]();
      /** @type {number} */
      var local$$43178 = (local$$43158[clientX] - local$$43165[left]) / local$$43165[width];
      /** @type {number} */
      var local$$43191 = (local$$43158[clientY] - local$$43165[top]) / local$$43165[height];
      local$$43193[set](local$$43178 * 2 - 1, -(local$$43191 * 2) + 1);
      local$$43208[setFromCamera](local$$43193, local$$41288);
      var local$$43219 = local$$43208[intersectObjects](local$$43159, true);
      return local$$43219[0] ? local$$43219[0] : false;
    }
    THREE[Object3D][call](this);
    local$$41289 = local$$41289 !== undefined ? local$$41289 : document;
    this[object] = undefined;
    /** @type {boolean} */
    this[visible] = false;
    /** @type {null} */
    this[translationSnap] = null;
    /** @type {null} */
    this[rotationSnap] = null;
    this[space] = world;
    /** @type {number} */
    this[size] = 1;
    /** @type {null} */
    this[axis] = null;
    var local$$41294 = this;
    var local$$41331 = translate;
    /** @type {boolean} */
    var local$$41299 = false;
    var local$$43292 = XY;
    var local$$41330 = {
      "translate" : new THREE.TransformGizmoTranslate,
      "rotate" : new THREE.TransformGizmoRotate,
      "scale" : new THREE.TransformGizmoScale
    };
    var local$$43302;
    for (local$$43302 in local$$41330) {
      var local$$43305 = local$$41330[local$$43302];
      /** @type {boolean} */
      local$$43305[visible] = local$$43302 === local$$41331;
      this[add](local$$43305);
    }
    var local$$41378 = {
      type : change
    };
    var local$$41454 = {
      type : mouseDown
    };
    var local$$43137 = {
      type : mouseUp,
      mode : local$$41331
    };
    var local$$43109 = {
      type : objectChange
    };
    var local$$43208 = new THREE.Raycaster;
    var local$$43193 = new THREE.Vector2;
    var local$$41676 = new THREE.Vector3;
    var local$$41589 = new THREE.Vector3;
    var local$$42344 = new THREE.Vector3;
    var local$$42380 = new THREE.Vector3;
    /** @type {number} */
    var local$$42129 = 1;
    var local$$42331 = new THREE.Matrix4;
    var local$$41473 = new THREE.Vector3;
    var local$$41573 = new THREE.Matrix4;
    var local$$42304 = new THREE.Vector3;
    var local$$42416 = new THREE.Quaternion;
    var local$$42648 = new THREE.Vector3(1, 0, 0);
    var local$$42674 = new THREE.Vector3(0, 1, 0);
    var local$$42700 = new THREE.Vector3(0, 0, 1);
    var local$$42439 = new THREE.Quaternion;
    var local$$42499 = new THREE.Quaternion;
    var local$$42670 = new THREE.Quaternion;
    var local$$42696 = new THREE.Quaternion;
    var local$$42426 = new THREE.Quaternion;
    var local$$41506 = new THREE.Vector3;
    var local$$41518 = new THREE.Vector3;
    var local$$41530 = new THREE.Matrix4;
    var local$$41554 = new THREE.Matrix4;
    var local$$41569 = new THREE.Vector3;
    var local$$41482 = new THREE.Vector3;
    var local$$43425 = new THREE.Euler;
    var local$$41542 = new THREE.Matrix4;
    var local$$41477 = new THREE.Vector3;
    var local$$43435 = new THREE.Euler;
    local$$41289[addEventListener](mousedown, local$$41385, false);
    local$$41289[addEventListener](touchstart, local$$41385, false);
    local$$41289[addEventListener](mousemove, local$$41291, false);
    local$$41289[addEventListener](touchmove, local$$41291, false);
    local$$41289[addEventListener](mousemove, local$$41610, false);
    local$$41289[addEventListener](touchmove, local$$41610, false);
    local$$41289[addEventListener](mouseup, local$$43114, false);
    local$$41289[addEventListener](mouseout, local$$43114, false);
    local$$41289[addEventListener](touchend, local$$43114, false);
    local$$41289[addEventListener](touchcancel, local$$43114, false);
    local$$41289[addEventListener](touchleave, local$$43114, false);
    /**
     * @return {undefined}
     */
    this[dispose] = function() {
      local$$41289[removeEventListener](mousedown, local$$41385);
      local$$41289[removeEventListener](touchstart, local$$41385);
      local$$41289[removeEventListener](mousemove, local$$41291);
      local$$41289[removeEventListener](touchmove, local$$41291);
      local$$41289[removeEventListener](mousemove, local$$41610);
      local$$41289[removeEventListener](touchmove, local$$41610);
      local$$41289[removeEventListener](mouseup, local$$43114);
      local$$41289[removeEventListener](mouseout, local$$43114);
      local$$41289[removeEventListener](touchend, local$$43114);
      local$$41289[removeEventListener](touchcancel, local$$43114);
      local$$41289[removeEventListener](touchleave, local$$43114);
    };
    /**
     * @param {?} local$$43613
     * @return {undefined}
     */
    this[attach] = function(local$$43613) {
      this[object] = local$$43613;
      /** @type {boolean} */
      this[visible] = true;
      this[update]();
    };
    /**
     * @return {undefined}
     */
    this[detach] = function() {
      this[object] = undefined;
      /** @type {boolean} */
      this[visible] = false;
      /** @type {null} */
      this[axis] = null;
    };
    /**
     * @param {string} local$$43663
     * @return {undefined}
     */
    this[setMode] = function(local$$43663) {
      local$$41331 = local$$43663 ? local$$43663 : local$$41331;
      if (local$$41331 === scale) {
        local$$41294[space] = local;
      }
      var local$$43681;
      for (local$$43681 in local$$41330) {
        /** @type {boolean} */
        local$$41330[local$$43681][visible] = local$$43681 === local$$41331;
      }
      this[update]();
      local$$41294[dispatchEvent](local$$41378);
    };
    /**
     * @param {?} local$$43710
     * @return {undefined}
     */
    this[setTranslationSnap] = function(local$$43710) {
      local$$41294[translationSnap] = local$$43710;
    };
    /**
     * @param {?} local$$43724
     * @return {undefined}
     */
    this[setRotationSnap] = function(local$$43724) {
      local$$41294[rotationSnap] = local$$43724;
    };
    /**
     * @param {?} local$$43738
     * @return {undefined}
     */
    this[setSize] = function(local$$43738) {
      local$$41294[size] = local$$43738;
      this[update]();
      local$$41294[dispatchEvent](local$$41378);
    };
    /**
     * @param {?} local$$43762
     * @return {undefined}
     */
    this[setSpace] = function(local$$43762) {
      local$$41294[space] = local$$43762;
      this[update]();
      local$$41294[dispatchEvent](local$$41378);
    };
    /**
     * @return {undefined}
     */
    this[update] = function() {
      if (local$$41294[object] === undefined) {
        return;
      }
      local$$41294[object][updateMatrixWorld]();
      local$$41482[setFromMatrixPosition](local$$41294[object][matrixWorld]);
      local$$43425[setFromRotationMatrix](local$$41573[extractRotation](local$$41294[object][matrixWorld]));
      local$$41288[updateMatrixWorld]();
      local$$41477[setFromMatrixPosition](local$$41288[matrixWorld]);
      local$$43435[setFromRotationMatrix](local$$41573[extractRotation](local$$41288[matrixWorld]));
      /** @type {number} */
      local$$42129 = local$$41482[distanceTo](local$$41477) / 6 * local$$41294[size];
      this[position][copy](local$$41482);
      this[scale][set](local$$42129, local$$42129, local$$42129);
      local$$41473[copy](local$$41477)[sub](local$$41482)[normalize]();
      if (local$$41294[space] === local) {
        local$$41330[local$$41331][update](local$$43425, local$$41473);
      } else {
        if (local$$41294[space] === world) {
          local$$41330[local$$41331][update](new THREE.Euler, local$$41473);
        }
      }
      local$$41330[local$$41331][highlight](local$$41294[axis]);
    };
  };
  THREE[TransformControls][prototype] = Object[create](THREE[Object3D][prototype]);
  THREE[TransformControls][prototype][constructor] = THREE[TransformControls];
})();
/**
 * @return {undefined}
 */
LSJSkyBox = function() {
  this[scene] = undefined;
  this[camera] = undefined;
  this[cubeMesh] = undefined;
  /** @type {!Array} */
  this[textures] = [resource/skybox/sky_0.jpg, resource/skybox/sky_1.jpg, resource/skybox/sky_5.jpg, resource/skybox/sky_4.jpg, resource/skybox/sky_3.jpg, resource/skybox/sky_2.jpg];
};
/** @type {function(): undefined} */
LSJSkyBox[prototype][constructor] = LSJSkyBox;
/**
 * @param {?} local$$44035
 * @param {?} local$$44036
 * @return {undefined}
 */
LSJSkyBox[prototype][loadSkyBox] = function(local$$44035, local$$44036) {
  var local$$44048 = (new THREE.CubeTextureLoader)[load](this[textures]);
  local$$44048[format] = THREE[RGBFormat];
  getScene()[background] = local$$44048;
};
/**
 * @param {?} local$$44075
 * @return {undefined}
 */
LSJSkyBox[prototype][setFrontTexture] = function(local$$44075) {
  this[textures][3] = local$$44075;
  this[loadSkyBox]();
};
/**
 * @param {?} local$$44099
 * @return {undefined}
 */
LSJSkyBox[prototype][setBackTexture] = function(local$$44099) {
  this[textures][2] = local$$44099;
  this[loadSkyBox]();
};
/**
 * @param {?} local$$44123
 * @return {undefined}
 */
LSJSkyBox[prototype][setUpTexture] = function(local$$44123) {
  this[textures][4] = local$$44123;
  this[loadSkyBox]();
};
/**
 * @param {?} local$$44147
 * @return {undefined}
 */
LSJSkyBox[prototype][setDownTexture] = function(local$$44147) {
  this[textures][5] = local$$44147;
  this[loadSkyBox]();
};
/**
 * @param {?} local$$44171
 * @return {undefined}
 */
LSJSkyBox[prototype][setRightTexture] = function(local$$44171) {
  this[textures][1] = local$$44171;
  this[loadSkyBox]();
};
/**
 * @param {?} local$$44195
 * @return {undefined}
 */
LSJSkyBox[prototype][setLiftTexture] = function(local$$44195) {
  this[textures][0] = local$$44195;
  this[loadSkyBox]();
};
/**
 * @param {(boolean|number|string)} local$$44219
 * @param {(boolean|number|string)} local$$44220
 * @return {undefined}
 */
LSJSkyBox[prototype][onSceneResize] = function(local$$44219, local$$44220) {
  if (this[camera]) {
    /** @type {number} */
    this[camera][aspect] = local$$44219 / local$$44220;
    this[camera][updateProjectionMatrix]();
  }
};
var Detector = {
  canvas : !!window[CanvasRenderingContext2D],
  webgl : function() {
    try {
      var local$$44262 = document[createElement](canvas);
      return !!(window[WebGLRenderingContext] && (local$$44262[getContext](webgl) || local$$44262[getContext](experimental-webgl)));
    } catch (local$$44285) {
      return false;
    }
  }(),
  workers : !!window[Worker],
  fileapi : window[File] && window[FileReader] && window[FileList] && window[Blob],
  getWebGLErrorMessage : function() {
    var local$$44324 = document[createElement](div);
    local$$44324[id] = webgl-error-message;
    local$$44324[style][fontFamily] = monospace;
    local$$44324[style][fontSize] = 13px;
    local$$44324[style][fontWeight] = normal;
    local$$44324[style][textAlign] = center;
    local$$44324[style][background] = #fff;
    local$$44324[style][color] = #000;
    local$$44324[style][padding] = 1.5em;
    local$$44324[style][width] = 400px;
    local$$44324[style][margin] = 5em auto 0;
    if (!this[webgl]) {
      local$$44324[innerHTML] = window[WebGLRenderingContext] ? [Your graphics card does not seem to support <a href="http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation" style="color:#000">WebGL</a>.<br />, Find out how to get it <a href="http://get.webgl.org/" style="color:#000">here</a>.][join](
) : [Your browser does not seem to support <a href="http://khronos.org/webgl/wiki/Getting_a_WebGL_Implementation" style="color:#000">WebGL</a>.<br/>, Find out how to get it <a href="http://get.webgl.org/" style="color:#000">here</a>.][join](
);
    }
    return local$$44324;
  },
  addGetWebGLMessage : function(local$$44464) {
    var local$$44466;
    var local$$44468;
    var local$$44470;
    local$$44464 = local$$44464 || {};
    local$$44466 = local$$44464[parent] !== undefined ? local$$44464[parent] : document[body];
    local$$44468 = local$$44464[id] !== undefined ? local$$44464[id] : oldie;
    local$$44470 = Detector[getWebGLErrorMessage]();
    local$$44470[id] = local$$44468;
    local$$44466[appendChild](local$$44470);
  }
};
if (typeof module === object) {
  module[exports] = Detector;
}
/**
 * @return {undefined}
 */
LSJStyle = function() {
  this[type] = none;
  /** @type {boolean} */
  this[depthTestUsed] = true;
};
/** @type {function(): undefined} */
LSJStyle[prototype][constructor] = LSJStyle;
/**
 * @return {?}
 */
LSJStyle[prototype][getType] = function() {
  return this[type];
};
/**
 * @return {undefined}
 */
LSJTextStyle = function() {
  this[fontName] = STHeiti;
  /** @type {number} */
  this[fontSize] = 32;
  /** @type {number} */
  this[scale] = 1;
  /** @type {boolean} */
  this[bold] = true;
  /** @type {boolean} */
  this[italic] = false;
  /** @type {boolean} */
  this[outlineVisible] = true;
  this[fillColor] = new THREE.Color;
  this[outlineColor] = new THREE.Color;
  this[outlineColor][setRGB](0, 0, 0);
  /** @type {number} */
  this[strokeWidth] = 1;
};
LSJTextStyle[prototype] = Object[create](LSJStyle[prototype]);
/** @type {function(): undefined} */
LSJTextStyle[prototype][constructor] = LSJTextStyle;
/**
 * @return {?}
 */
LSJTextStyle[prototype][getFontName] = function() {
  return this[fontName];
};
/**
 * @param {?} local$$44694
 * @return {undefined}
 */
LSJTextStyle[prototype][setFontName] = function(local$$44694) {
  this[fontName] = local$$44694;
};
/**
 * @return {?}
 */
LSJTextStyle[prototype][getFontSize] = function() {
  return this[fontSize];
};
/**
 * @param {?} local$$44726
 * @return {undefined}
 */
LSJTextStyle[prototype][setFontSize] = function(local$$44726) {
  this[fontSize] = local$$44726;
};
/**
 * @return {?}
 */
LSJTextStyle[prototype][getFillColor] = function() {
  return this[fillColor];
};
/**
 * @param {?} local$$44758
 * @return {undefined}
 */
LSJTextStyle[prototype][setFillColor] = function(local$$44758) {
  this[fillColor][copy](local$$44758);
};
/**
 * @return {?}
 */
LSJTextStyle[prototype][getOutlineColor] = function() {
  return this[outlineColor];
};
/**
 * @param {?} local$$44793
 * @return {undefined}
 */
LSJTextStyle[prototype][setOutlineColor] = function(local$$44793) {
  this[outlineColor][copy](local$$44793);
};
/**
 * @return {?}
 */
LSJTextStyle[prototype][getStrokeWidth] = function() {
  return this[strokeWidth];
};
/**
 * @param {?} local$$44828
 * @return {undefined}
 */
LSJTextStyle[prototype][setStrokeWidth] = function(local$$44828) {
  this[strokeWidth] = local$$44828;
};
/**
 * @return {undefined}
 */
LSJMarkerStyle = function() {
  LSJStyle[call](this);
  this[type] = MarkerStyle;
  this[iconPath] = undefined;
  this[iconColor] = new THREE.Color;
  /** @type {number} */
  this[iconSize] = 10;
  /** @type {boolean} */
  this[iconVisible] = true;
  /** @type {boolean} */
  this[textVisible] = true;
  /** @type {boolean} */
  this[iconFixedSize] = true;
  this[textStyle] = new LSJTextStyle;
};
LSJMarkerStyle[prototype] = Object[create](LSJStyle[prototype]);
/** @type {function(): undefined} */
LSJMarkerStyle[prototype][constructor] = LSJMarkerStyle;
/**
 * @param {?} local$$44927
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setIconColor] = function(local$$44927) {
  this[iconColor] = local$$44927;
};
/**
 * @return {?}
 */
LSJMarkerStyle[prototype][getIconPath] = function() {
  return this[iconPath];
};
/**
 * @param {?} local$$44959
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setIconPath] = function(local$$44959) {
  this[iconPath] = local$$44959;
};
/**
 * @return {?}
 */
LSJMarkerStyle[prototype][getIconScale] = function() {
  return this[iconScale];
};
/**
 * @param {?} local$$44991
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setIconScale] = function(local$$44991) {
  this[iconScale] = local$$44991;
};
/**
 * @param {?} local$$45008
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setStyle] = function(local$$45008) {
  this[style] = local$$45008;
};
/**
 * @param {?} local$$45025
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setTextStyle] = function(local$$45025) {
  this[textStyle] = local$$45025;
};
/**
 * @return {?}
 */
LSJMarkerStyle[prototype][getTextStyle] = function() {
  return this[textStyle];
};
/**
 * @param {?} local$$45057
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setIconVisible] = function(local$$45057) {
  this[iconVisible] = local$$45057;
};
/**
 * @return {?}
 */
LSJMarkerStyle[prototype][getIconVisible] = function() {
  return this[iconVisible];
};
/**
 * @param {?} local$$45089
 * @return {undefined}
 */
LSJMarkerStyle[prototype][setTextVisible] = function(local$$45089) {
  this[textVisible] = local$$45089;
};
/**
 * @return {?}
 */
LSJMarkerStyle[prototype][getTextVisible] = function() {
  return this[textVisible];
};
/**
 * @return {undefined}
 */
LSJIconStyle = function() {
  LSJStyle[call](this);
  this[type] = IconStyle;
  this[iconPath] = ;
  this[iconColor] = new THREE.Color;
  /** @type {number} */
  this[iconScale] = 1;
};
LSJIconStyle[prototype] = Object[create](LSJStyle[prototype]);
/** @type {function(): undefined} */
LSJIconStyle[prototype][constructor] = LSJIconStyle;
/**
 * @return {?}
 */
LSJIconStyle[prototype][getIconColor] = function() {
  return this[iconColor];
};
/**
 * @param {?} local$$45196
 * @return {undefined}
 */
LSJIconStyle[prototype][setIconColor] = function(local$$45196) {
  this[iconColor] = local$$45196;
};
/**
 * @return {?}
 */
LSJIconStyle[prototype][getIconPath] = function() {
  return this[iconPath];
};
/**
 * @param {?} local$$45228
 * @return {undefined}
 */
LSJIconStyle[prototype][setIconPath] = function(local$$45228) {
  this[iconPath] = local$$45228;
};
/**
 * @return {?}
 */
LSJIconStyle[prototype][getIconScale] = function() {
  return this[iconScale];
};
/**
 * @param {?} local$$45260
 * @return {undefined}
 */
LSJIconStyle[prototype][setIconScale] = function(local$$45260) {
  this[iconScale] = local$$45260;
};
/**
 * @return {undefined}
 */
LSJGeometry = function() {
  this[meshGroup] = new THREE.Group;
  /** @type {number} */
  this[id] = 0;
  this[name] = ;
  this[description] = ;
  this[type] = none;
  /** @type {null} */
  this[style] = null;
  /** @type {boolean} */
  this[visible] = true;
};
/** @type {function(): undefined} */
LSJGeometry[prototype][constructor] = LSJGeometry;
/**
 * @return {undefined}
 */
LSJGeometry[prototype][dispose] = function() {
  for (; this[meshGroup][children][length] > 0;) {
    var local$$45361 = this[meshGroup][children][0];
    this[meshGroup][remove](local$$45361);
  }
};
/**
 * @return {?}
 */
LSJGeometry[prototype][getName] = function() {
  return this[name];
};
/**
 * @return {?}
 */
LSJGeometry[prototype][getType] = function() {
  return this[type];
};
/**
 * @return {?}
 */
LSJGeometry[prototype][getId] = function() {
  return this[id];
};
/**
 * @return {?}
 */
LSJGeometry[prototype][getStyle] = function() {
  return this[style];
};
/**
 * @return {undefined}
 */
LSJGeoMarker = function() {
  LSJGeometry[call](this);
  this[type] = GeoMarker;
  this[strIconPath] = ;
  this[position] = new THREE.Vector3(0, 0, 0);
  this[billboard] = undefined;
  /** @type {boolean} */
  this[needUpdate] = true;
  this[screenRect] = new LSJRectangle(0, 0, 0, 0);
  /** @type {boolean} */
  this[bIsNameVisble] = true;
  /** @type {number} */
  this[actualAspect] = 1;
};
LSJGeoMarker[prototype] = Object[create](LSJGeometry[prototype]);
/** @type {function(): undefined} */
LSJGeoMarker[prototype][constructor] = LSJGeoMarker;
/**
 * @param {!Object} local$$45533
 * @return {undefined}
 */
LSJGeoMarker[prototype][setStyle] = function(local$$45533) {
  if (local$$45533 == null) {
    return;
  }
  /** @type {!Object} */
  this[style] = local$$45533;
  /** @type {boolean} */
  this[needUpdate] = true;
};
/**
 * @param {?} local$$45562
 * @return {undefined}
 */
LSJGeoMarker[prototype][setName] = function(local$$45562) {
  this[name] = local$$45562;
  /** @type {boolean} */
  this[needUpdate] = true;
};
/**
 * @return {undefined}
 */
LSJGeoMarker[prototype][update] = function() {
  this[dispose]();
  if (this[style][iconPath] != undefined) {
    this[strIconPath] = this[style][iconPath];
    /** @type {!Image} */
    var local$$45610 = new Image;
    local$$45610[src] = this[style][iconPath];
    var local$$45623 = this;
    /**
     * @return {undefined}
     */
    local$$45610[onload] = function() {
      var local$$45629 = undefined;
      if (local$$45623[name] !=  && local$$45623[bIsNameVisble]) {
        var local$$45648 = local$$45623[style][getTextStyle]();
        local$$45629 = writeTextAndImgToCanvas(local$$45623[name], local$$45610, local$$45648);
        /** @type {number} */
        local$$45623[actualAspect] = local$$45629[width] / local$$45629[height];
      } else {
        local$$45629 = document[createElement](canvas);
        /** @type {number} */
        local$$45629[width] = 128 * local$$45610[height] / local$$45610[width];
        /** @type {number} */
        local$$45629[height] = 128;
        if (local$$45623[name] == ) {
          /** @type {number} */
          local$$45623[actualAspect] = local$$45629[width] / local$$45629[height];
        }
      }
      var local$$45726 = local$$45629[getContext](2d);
      local$$45726[drawImage](local$$45610, 0, 0, local$$45629[height] * local$$45610[width] / local$$45610[height], local$$45629[height]);
      var local$$45756 = new THREE.SpriteMaterial({
        depthTest : false,
        map : new THREE.CanvasTexture(local$$45629)
      });
      local$$45623[billboard] = new LSJBillboard(local$$45756, local$$45623);
      local$$45623[billboard][position][copy](local$$45623[position]);
      local$$45623[meshGroup][add](local$$45623[billboard]);
      /** @type {boolean} */
      local$$45623[needUpdate] = false;
    };
  } else {
    var local$$45807 = this[style][getTextStyle]();
    var local$$45814 = writeTextAndImgToCanvas(this[name], null, local$$45807);
    var local$$45821 = new THREE.SpriteMaterial({
      map : new THREE.CanvasTexture(local$$45814)
    });
    /** @type {number} */
    this[actualAspect] = local$$45814[width] / local$$45814[height];
    this[billboard] = new LSJBillboard(local$$45821, this);
    this[billboard][position][copy](this[position]);
    this[meshGroup][add](this[billboard]);
    /** @type {boolean} */
    this[needUpdate] = false;
  }
};
/**
 * @param {?} local$$45884
 * @param {?} local$$45885
 * @param {?} local$$45886
 * @return {undefined}
 */
LSJGeoMarker[prototype][setPosition] = function(local$$45884, local$$45885, local$$45886) {
  this[position][x] = local$$45884;
  this[position][y] = local$$45885;
  this[position][z] = local$$45886;
  if (this[billboard] != undefined) {
    this[billboard][position][copy](this[position]);
  }
};
/**
 * @return {?}
 */
LSJGeoMarker[prototype][getPosition] = function() {
  return this[postion];
};
/**
 * @return {?}
 */
LSJGeoMarker[prototype][getScreenRect] = function() {
  return this[screenRect];
};
/**
 * @param {?} local$$45972
 * @return {undefined}
 */
LSJGeoMarker[prototype][setNameVisble] = function(local$$45972) {
  if (this[bIsNameVisble] != local$$45972) {
    /** @type {boolean} */
    this[needUpdate] = true;
  }
  this[bIsNameVisble] = local$$45972;
};
/**
 * @return {?}
 */
LSJGeoMarker[prototype][getNameVisble] = function() {
  return this[bIsNameVisble];
};
/**
 * @param {string} local$$46017
 * @return {undefined}
 */
LSJGeoMarker[prototype][render] = function(local$$46017) {
  if (this[needUpdate]) {
    this[update]();
  }
  if (this[billboard] != undefined) {
    this[billboard][camera] = getCamera();
    this[billboard][updateMatrixWorld]();
    local$$46017[controlCamera][updateMatrixWorld]();
    local$$46017[controlCamera][updateProjectionMatrix]();
    var local$$46073 = new THREE.Vector3(0, 0, 0);
    var local$$46080 = new THREE.Vector3(0, 0, 0);
    local$$46080[copy](this[billboard][position]);
    /** @type {number} */
    var local$$46094 = 1;
    /** @type {number} */
    var local$$46097 = 1;
    var local$$46107 = getCamera()[position][distanceTo](local$$46080);
    if (local$$46107 > local$$46017[boundingSphere][radius] && local$$46017[boundingSphere][radius] != 0) {
      var local$$46163 = getCamera()[position][clone]()[add](getCamera()[position][clone]()[sub](local$$46080)[normalize]()[multiplyScalar](local$$46017[boundingSphere][radius]));
      var local$$46176 = local$$46163[clone]()[project](local$$46017[controlCamera]);
      /** @type {number} */
      var local$$46195 = (local$$46176[x] + 1) / 2 * local$$46017[controlRender][domElement][clientWidth];
      var local$$46202 = new THREE.Vector3(0, 1, 0);
      local$$46202[applyQuaternion](local$$46017[controlCamera][quaternion]);
      var local$$46230 = local$$46163[clone]()[add](local$$46202)[project](local$$46017[controlCamera]);
      /** @type {number} */
      var local$$46250 = -(local$$46176[y] - 1) / 2 * local$$46017[controlRender][domElement][clientHeight];
      /** @type {number} */
      var local$$46270 = -(local$$46230[y] - 1) / 2 * local$$46017[controlRender][domElement][clientHeight];
      /** @type {number} */
      var local$$46279 = 1 / Math[abs](local$$46250 - local$$46270);
      local$$46094 = Math[abs](local$$46279);
      local$$46176 = local$$46080[clone]()[project](local$$46017[controlCamera]);
      local$$46202 = new THREE.Vector3(0, 1, 0);
      local$$46202[applyQuaternion](local$$46017[controlCamera][quaternion]);
      local$$46230 = local$$46080[clone]()[add](local$$46202)[project](local$$46017[controlCamera]);
      /** @type {number} */
      local$$46250 = -(local$$46176[y] - 1) / 2 * local$$46017[controlRender][domElement][clientHeight];
      /** @type {number} */
      local$$46270 = -(local$$46230[y] - 1) / 2 * local$$46017[controlRender][domElement][clientHeight];
      /** @type {number} */
      var local$$46382 = 1 / Math[abs](local$$46250 - local$$46270);
      if (local$$46382 > 2 * local$$46279) {
        /** @type {number} */
        local$$46094 = local$$46382 / 2;
      }
      /** @type {number} */
      local$$46097 = local$$46279 / local$$46382;
    } else {
      local$$46176 = local$$46080[clone]()[project](local$$46017[controlCamera]);
      local$$46202 = new THREE.Vector3(0, 1, 0);
      local$$46202[applyQuaternion](local$$46017[controlCamera][quaternion]);
      local$$46230 = local$$46080[clone]()[add](local$$46202)[project](local$$46017[controlCamera]);
      /** @type {number} */
      local$$46250 = -(local$$46176[y] - 1) / 2 * local$$46017[controlRender][domElement][clientHeight];
      /** @type {number} */
      local$$46270 = -(local$$46230[y] - 1) / 2 * local$$46017[controlRender][domElement][clientHeight];
      /** @type {number} */
      local$$46094 = 1 / Math[abs](local$$46250 - local$$46270);
      /** @type {number} */
      local$$46195 = (local$$46176[x] + 1) / 2 * local$$46017[controlRender][domElement][clientWidth];
    }
    if (this[style][iconPath] != undefined) {
      /** @type {number} */
      this[billboard][scale][x] = this[billboard][scale][y] = this[billboard][scale][z] = this[style][iconSize] * local$$46094;
      /** @type {number} */
      this[screenRect][left] = local$$46195;
      /** @type {number} */
      this[screenRect][bottom] = local$$46250;
      /** @type {number} */
      this[screenRect][top] = local$$46250 + this[style][iconSize] * local$$46097;
    } else {
      var local$$46602 = this[style][getTextStyle]();
      /** @type {number} */
      this[billboard][scale][x] = this[billboard][scale][y] = this[billboard][scale][z] = local$$46602[getFontSize]() * local$$46094;
      /** @type {number} */
      this[screenRect][left] = local$$46195;
      /** @type {number} */
      this[screenRect][bottom] = local$$46250;
      /** @type {number} */
      this[screenRect][top] = local$$46250 + local$$46602[getFontSize]() * local$$46097;
    }
    /** @type {number} */
    var local$$46686 = this[screenRect][top] - this[screenRect][bottom];
    /** @type {number} */
    var local$$46692 = local$$46686 * this[actualAspect];
    this[screenRect][right] = this[screenRect][left] + local$$46692;
    if (local$$46017 != undefined) {
      local$$46017[billboards][push](this[billboard]);
    }
  }
};
/**
 * @return {undefined}
 */
LSJMath = function() {
};
/** @type {function(): undefined} */
LSJMath[prototype][constructor] = LSJMath;
/**
 * @param {?} local$$46747
 * @param {?} local$$46748
 * @return {undefined}
 */
LSJMath[expandSphere] = function(local$$46747, local$$46748) {
  if (local$$46748[empty]()) {
    return;
  }
  if (local$$46747[empty]()) {
    local$$46747[set](local$$46748[center], local$$46748[radius]);
    return;
  }
  var local$$46779 = new THREE.Vector3;
  local$$46779[subVectors](local$$46747[center], local$$46748[center]);
  var local$$46796 = local$$46779[length]();
  if (local$$46796 + local$$46748[radius] <= local$$46747[radius]) {
    return;
  }
  if (local$$46796 + local$$46747[radius] <= local$$46748[radius]) {
    local$$46747[set](local$$46748[center], local$$46748[radius]);
    return;
  }
  /** @type {number} */
  var local$$46843 = (local$$46747[radius] + local$$46796 + local$$46748[radius]) * .5;
  /** @type {number} */
  var local$$46850 = (local$$46843 - local$$46747[radius]) / local$$46796;
  var local$$46872 = new THREE.Vector3(local$$46747[center][x], local$$46747[center][y], local$$46747[center][z]);
  local$$46872[x] += (local$$46748[center][x] - local$$46747[center][x]) * local$$46850;
  local$$46872[y] += (local$$46748[center][y] - local$$46747[center][y]) * local$$46850;
  local$$46872[x] += (local$$46748[center][z] - local$$46747[center][z]) * local$$46850;
  local$$46747[set](local$$46872, local$$46843);
};
/**
 * @param {?} local$$46943
 * @param {?} local$$46944
 * @return {?}
 */
LSJMath[computeTowVecDist] = function(local$$46943, local$$46944) {
  /** @type {number} */
  var local$$46953 = local$$46943[x] - local$$46944[x];
  /** @type {number} */
  var local$$46962 = local$$46943[y] - local$$46944[y];
  /** @type {number} */
  var local$$46971 = local$$46943[z] - local$$46944[z];
  return Math[sqrt](local$$46953 * local$$46953 + local$$46962 * local$$46962 + local$$46971 * local$$46971);
};
/**
 * @param {?} local$$46990
 * @param {?} local$$46991
 * @return {?}
 */
LSJMath[computeTowVecDistSquare] = function(local$$46990, local$$46991) {
  /** @type {number} */
  var local$$47000 = local$$46990[x] - local$$46991[x];
  /** @type {number} */
  var local$$47009 = local$$46990[y] - local$$46991[y];
  /** @type {number} */
  var local$$47018 = local$$46990[z] - local$$46991[z];
  return local$$47000 * local$$47000 + local$$47009 * local$$47009 + local$$47018 * local$$47018;
};
/**
 * @param {?} local$$47033
 * @param {?} local$$47034
 * @return {?}
 */
LSJMath[computeDistFromEye] = function(local$$47033, local$$47034) {
  var local$$47040 = local$$47033[applyMatrix4](local$$47034);
  return local$$47040[length]();
};
/**
 * @param {?} local$$47054
 * @return {?}
 */
LSJMath[Er] = function(local$$47054) {
  if (!LSJMath[us](local$$47054[width]) || !LSJMath[us](local$$47054[height])) {
    var local$$47079 = document[createElement](canvas);
    local$$47079[width] = LSJMath.Fs(local$$47054[width]);
    local$$47079[height] = LSJMath.Fs(local$$47054[height]);
    local$$47079[getContext](2d)[drawImage](local$$47054, 0, 0, local$$47054[width], local$$47054[height], 0, 0, local$$47079[width], local$$47079[height]);
    return local$$47079;
  }
  return local$$47054;
};
/**
 * @param {number} local$$47140
 * @return {?}
 */
LSJMath[us] = function(local$$47140) {
  return 0 === (local$$47140 & local$$47140 - 1);
};
/**
 * @param {number} local$$47155
 * @return {?}
 */
LSJMath[Fs] = function(local$$47155) {
  --local$$47155;
  /** @type {number} */
  var local$$47160 = 1;
  for (; 32 > local$$47160; local$$47160 = local$$47160 << 1) {
    /** @type {number} */
    local$$47155 = local$$47155 | local$$47155 >> local$$47160;
  }
  return local$$47155 + 1;
};
/**
 * @param {?} local$$47185
 * @param {?} local$$47186
 * @return {?}
 */
LSJMath[mulVec3Vec4] = function(local$$47185, local$$47186) {
  return local$$47185[x] * local$$47186[x] + local$$47185[y] * local$$47186[y] + local$$47185[z] * local$$47186[z] + local$$47186[w];
};
/**
 * @param {?} local$$47223
 * @return {?}
 */
LSJMath[isZeroVec2] = function(local$$47223) {
  return local$$47223[x] == 0 && local$$47223[y] == 0;
};
/**
 * @param {?} local$$47244
 * @return {?}
 */
LSJMath[isZeroVec3] = function(local$$47244) {
  return local$$47244[x] == 0 && local$$47244[y] == 0 && local$$47244[z] == 0;
};
/**
 * @param {?} local$$47271
 * @param {?} local$$47272
 * @return {?}
 */
LSJMath[computeSpherePixelSize] = function(local$$47271, local$$47272) {
  return Math[abs](local$$47271[radius] / LSJMath[mulVec3Vec4](local$$47271[center], local$$47272));
};
/**
 * @param {?} local$$47297
 * @param {?} local$$47298
 * @param {?} local$$47299
 * @return {?}
 */
LSJMath[computePixelSizeVector] = function(local$$47297, local$$47298, local$$47299) {
  var local$$47304 = local$$47297[z];
  var local$$47309 = local$$47297[w];
  var local$$47314 = local$$47298[elements];
  var local$$47319 = local$$47299[elements];
  /** @type {number} */
  var local$$47326 = local$$47314[0] * local$$47304 * .5;
  /** @type {number} */
  var local$$47339 = local$$47314[8] * local$$47304 * .5 + local$$47314[11] * local$$47304 * .5;
  var local$$47364 = new THREE.Vector3(local$$47319[0] * local$$47326 + local$$47319[2] * local$$47339, local$$47319[4] * local$$47326 + local$$47319[6] * local$$47339, local$$47319[8] * local$$47326 + local$$47319[10] * local$$47339);
  /** @type {number} */
  var local$$47371 = local$$47314[5] * local$$47309 * .5;
  /** @type {number} */
  var local$$47384 = local$$47314[9] * local$$47309 * .5 + local$$47314[11] * local$$47309 * .5;
  var local$$47409 = new THREE.Vector3(local$$47319[1] * local$$47371 + local$$47319[2] * local$$47384, local$$47319[5] * local$$47371 + local$$47319[6] * local$$47384, local$$47319[9] * local$$47371 + local$$47319[10] * local$$47384);
  var local$$47413 = local$$47314[11];
  var local$$47417 = local$$47314[15];
  var local$$47437 = new THREE.Vector4(local$$47319[2] * local$$47413, local$$47319[6] * local$$47413, local$$47319[10] * local$$47413, local$$47319[14] * local$$47413 + local$$47319[15] * local$$47417);
  /** @type {number} */
  var local$$47454 = .7071067811 / Math[sqrt](local$$47364[lengthSq]() + local$$47409[lengthSq]());
  local$$47437[multiplyScalar](local$$47454);
  return local$$47437;
};
/**
 * @param {number} local$$47469
 * @return {?}
 */
LSJMath[us] = function(local$$47469) {
  return 0 === (local$$47469 & local$$47469 - 1);
};
/**
 * @param {number} local$$47484
 * @return {?}
 */
LSJMath[Fs] = function(local$$47484) {
  --local$$47484;
  /** @type {number} */
  var local$$47489 = 1;
  for (; 32 > local$$47489; local$$47489 = local$$47489 << 1) {
    /** @type {number} */
    local$$47484 = local$$47484 | local$$47484 >> local$$47489;
  }
  return local$$47484 + 1;
};
/**
 * @param {number} local$$47514
 * @return {?}
 */
LSJMath[nextHighestPowerOfTwo_] = function(local$$47514) {
  --local$$47514;
  /** @type {number} */
  var local$$47519 = 1;
  for (; local$$47519 < 32; local$$47519 = local$$47519 << 1) {
    /** @type {number} */
    local$$47514 = local$$47514 | local$$47514 >> local$$47519;
  }
  return local$$47514 + 1;
};
/**
 * @return {undefined}
 */
LSJGeoModel = function() {
  LSJGeometry[call](this);
  this[type] = GeoModel;
  this[position] = new THREE.Vector3(0, 0, 0);
  this[boundingSphere] = new THREE.Sphere;
};
LSJGeoModel[prototype] = Object[create](LSJGeometry[prototype]);
/** @type {function(): undefined} */
LSJGeoModel[prototype][constructor] = LSJGeoModel;
/**
 * @return {undefined}
 */
LSJGeoModel[prototype][update] = function() {
};
/**
 * @param {?} local$$47615
 * @param {?} local$$47616
 * @param {?} local$$47617
 * @return {undefined}
 */
LSJGeoModel[prototype][setPosition] = function(local$$47615, local$$47616, local$$47617) {
  this[position][x] = local$$47615;
  this[position][y] = local$$47616;
  this[position][z] = local$$47617;
};
/**
 * @return {?}
 */
LSJGeoModel[prototype][getPosition] = function() {
  return this[postion];
};
/**
 * @param {string} local$$47668
 * @return {undefined}
 */
LSJGeoModel[prototype][load] = function(local$$47668) {
  /**
   * @param {?} local$$47670
   * @return {undefined}
   */
  var local$$47693 = function(local$$47670) {
    if (local$$47670[lengthComputable]) {
      /** @type {number} */
      var local$$47684 = local$$47670[loaded] / local$$47670[total] * 100;
      onProgressInfo(local$$47684);
    }
  };
  /**
   * @param {?} local$$47695
   * @return {undefined}
   */
  var local$$47699 = function(local$$47695) {
  };
  var local$$47703 = local$$47668.toString();
  var local$$47718 = local$$47703[slice](0, local$$47703[length] - 3) + mtl;
  var local$$47733 = local$$47718[substring](0, local$$47718[lastIndexOf](/) + 1);
  var local$$47750 = local$$47718[substring](local$$47718[lastIndexOf](/) + 1, local$$47718[length]);
  var local$$47752 = this;
  var local$$47756 = new THREE.MTLLoader;
  local$$47756[setPath](local$$47733);
  local$$47756[load](local$$47750, function(local$$47766) {
    local$$47766[preload]();
    var local$$47775 = new THREE.OBJLoader;
    local$$47775[setMaterials](local$$47766);
    local$$47775[load](local$$47703, function(local$$47785) {
      /** @type {boolean} */
      local$$47785[castShadow] = true;
      /** @type {boolean} */
      local$$47785[receiveShadow] = true;
      local$$47752[meshGroup][add](local$$47785);
      /** @type {number} */
      var local$$47808 = 0;
      for (; local$$47808 < local$$47785[children][length]; local$$47808++) {
        if (local$$47785[children][local$$47808][type] == Object3D) {
          /** @type {number} */
          var local$$47830 = 0;
          for (; local$$47830 < local$$47785[children][local$$47808][children][length]; local$$47830++) {
            var local$$47856 = local$$47785[children][local$$47808][children][local$$47830][geometry];
            /** @type {boolean} */
            local$$47785[children][local$$47808][children][local$$47830][castShadow] = true;
            /** @type {boolean} */
            local$$47785[children][local$$47808][children][local$$47830][receiveShadow] = true;
            if (local$$47856 != undefined && local$$47856[boundingSphere] === null) {
              local$$47856[computeBoundingSphere]();
            }
            if (local$$47856 != undefined) {
              local$$47856[rotateX](0);
              LSJMath[expandSphere](local$$47752[boundingSphere], local$$47856[boundingSphere]);
            }
          }
        } else {
          if (local$$47785[children][local$$47808][type] == Mesh) {
            local$$47856 = local$$47785[children][local$$47808][geometry];
            /** @type {boolean} */
            local$$47785[children][local$$47808][castShadow] = true;
            /** @type {boolean} */
            local$$47785[children][local$$47808][receiveShadow] = true;
            if (local$$47856 != undefined && local$$47856[boundingSphere] === null) {
              local$$47856[computeBoundingSphere]();
            }
            if (local$$47856 != undefined) {
              local$$47856[rotateX](0);
              LSJMath[expandSphere](local$$47752[boundingSphere], local$$47856[boundingSphere]);
            }
          }
        }
      }
    }, local$$47693, local$$47699);
  });
};
/**
 * @return {?}
 */
LSJGeoModel[prototype][getBoundingSphere] = function() {
  return this[boundingSphere];
};
/**
 * @return {undefined}
 */
LSJLayer = function() {
  /** @type {!Array} */
  this[geometrys] = [];
  this[meshGroup] = new THREE.Group;
  this[boundingSphere] = new THREE.Sphere;
  /** @type {number} */
  this[maxID] = 0;
  /** @type {number} */
  getScene()[curSendNode] = 0;
  /** @type {!Date} */
  this[date] = new Date;
  getScene()[lastTime] = this[date][getTime]();
  /** @type {number} */
  this[lastUpadeIndex] = 0;
};
/** @type {function(): undefined} */
LSJLayer[prototype][constructor] = LSJLayer;
/**
 * @return {undefined}
 */
LSJLayer[prototype][dispose] = function() {
  var local$$48116 = this[geometrys][length];
  /** @type {number} */
  var local$$48119 = 0;
  for (; local$$48119 < local$$48116; local$$48119++) {
    var local$$48128 = this[geometrys][local$$48119];
    /** @type {null} */
    local$$48128[meshGroup][attachObject] = null;
    this[meshGroup][remove](local$$48128[meshGroup]);
    if (local$$48128 != null) {
      local$$48128[dispose]();
    }
  }
  this[geometrys][slice](0, local$$48116);
  /** @type {!Array} */
  this[geometrys] = [];
};
/**
 * @return {undefined}
 */
LSJLayer[prototype][removeAll] = function() {
  var local$$48194 = this[geometrys][length];
  /** @type {number} */
  var local$$48197 = 0;
  for (; local$$48197 < local$$48194; local$$48197++) {
    var local$$48206 = this[geometrys][local$$48197];
    /** @type {null} */
    local$$48206[meshGroup][attachObject] = null;
    this[meshGroup][remove](local$$48206[meshGroup]);
    if (local$$48206 != null) {
      local$$48206[dispose]();
    }
  }
  this[geometrys][slice](0, local$$48194);
  /** @type {!Array} */
  this[geometrys] = [];
};
/**
 * @return {?}
 */
LSJLayer[prototype][getBoundingSphere] = function() {
  if (this[boundingSphere][empty]()) {
    var local$$48279 = this[geometrys][length];
    /** @type {number} */
    var local$$48282 = 0;
    for (; local$$48282 < local$$48279; local$$48282++) {
      var local$$48291 = this[geometrys][local$$48282];
      if (local$$48291 != undefined) {
        if (local$$48291[type] == GeoModel || local$$48291[type] == GeoModelLOD) {
          LSJMath[expandSphere](this[boundingSphere], local$$48291[getBoundingSphere]());
        }
      }
    }
  }
  return this[boundingSphere];
};
/**
 * @param {!Object} local$$48342
 * @return {undefined}
 */
LSJLayer[prototype][addGeometry] = function(local$$48342) {
  if (local$$48342 == null || local$$48342 == undefined) {
    return;
  }
  this[maxID]++;
  local$$48342[id] = this[maxID];
  local$$48342[layer] = this;
  this[geometrys][push](local$$48342);
  if (local$$48342[meshGroup] != null) {
    this[meshGroup][add](local$$48342[meshGroup]);
  }
};
/**
 * @param {?} local$$48406
 * @return {undefined}
 */
LSJLayer[prototype][removeGeometryByName] = function(local$$48406) {
  var local$$48412 = this[getGeometryByName](local$$48406);
  /** @type {null} */
  local$$48412[meshGroup][attachObject] = null;
  this[meshGroup][remove](local$$48412[meshGroup]);
  if (local$$48412 != null) {
    var local$$48443 = this[geometrys][indexOf](local$$48412);
    if (local$$48443 !== -1) {
      this[geometrys][splice](local$$48443, 1);
    }
    local$$48412[dispose]();
    return;
  }
};
/**
 * @param {?} local$$48477
 * @return {undefined}
 */
LSJLayer[prototype][removeGeometryByID] = function(local$$48477) {
  var local$$48483 = this[getGeometryByID](local$$48477);
  /** @type {null} */
  local$$48483[meshGroup][attachObject] = null;
  this[meshGroup][remove](local$$48483[meshGroup]);
  if (local$$48483 != null) {
    var local$$48514 = this[geometrys][indexOf](local$$48483);
    if (local$$48514 !== -1) {
      this[geometrys][splice](local$$48514, 1);
    }
    local$$48483[dispose]();
    return;
  }
};
/**
 * @param {?} local$$48548
 * @return {?}
 */
LSJLayer[prototype][getGeometryByName] = function(local$$48548) {
  var local$$48556 = this[geometrys][length];
  /** @type {number} */
  var local$$48559 = 0;
  for (; local$$48559 < local$$48556; local$$48559++) {
    var local$$48568 = this[geometrys][local$$48559];
    if (local$$48568 != null) {
      if (local$$48568[name] == local$$48548) {
        return local$$48568;
      }
    }
  }
  return null;
};
/**
 * @param {?} local$$48596
 * @return {?}
 */
LSJLayer[prototype][getGeometryByID] = function(local$$48596) {
  var local$$48604 = this[geometrys][length];
  /** @type {number} */
  var local$$48607 = 0;
  for (; local$$48607 < local$$48604; local$$48607++) {
    var local$$48616 = this[geometrys][local$$48607];
    if (local$$48616 != null) {
      if (local$$48616[id] == local$$48596) {
        return local$$48616;
      }
    }
  }
  return null;
};
/**
 * @param {number} local$$48644
 * @return {?}
 */
LSJLayer[prototype][getGeometryByIndex] = function(local$$48644) {
  var local$$48652 = this[geometrys][length];
  if (local$$48644 >= 0 && local$$48644 < local$$48652) {
    return this[geometrys][local$$48644];
  }
  return null;
};
/**
 * @param {?} local$$48678
 * @return {?}
 */
LSJLayer[prototype][render] = function(local$$48678) {
  /** @type {!Date} */
  var local$$48681 = new Date;
  getScene()[lastTime] = local$$48681[getTime]();
  var local$$48699 = this[geometrys][length];
  /** @type {number} */
  var local$$48702 = 0;
  /** @type {number} */
  var local$$48705 = 0;
  for (; local$$48705 < local$$48699; local$$48705++) {
    var local$$48714 = this[geometrys][local$$48705];
    if (local$$48714 != null) {
      if (local$$48714[type] == GeoMarker || local$$48714[type] == GeoLabel || local$$48714[type] == GeoModelLOD || local$$48714[type] == GeoPolygon) {
        if (local$$48714[type] == GeoLabel) {
          /** @type {!Date} */
          var local$$48752 = new Date;
          var local$$48758 = local$$48752[getTime]();
          if (local$$48758 - getScene()[lastTime] > 10 || this[lastUpadeIndex] > local$$48705) {
            local$$48714[render](local$$48678, false);
          } else {
            local$$48714[render](local$$48678, true);
            /** @type {number} */
            local$$48702 = local$$48705;
          }
        } else {
          local$$48714[render](local$$48678);
          if (getScene()[curSendNode] > 2) {
            break;
          }
        }
      }
    }
  }
  /** @type {number} */
  this[lastUpadeIndex] = local$$48702 == local$$48699 - 1 ? 0 : local$$48702;
  return null;
};
/**
 * @return {undefined}
 */
LSJPageLOD = function() {
  this[name] = ;
  this[type] = PageLOD;
  /** @type {boolean} */
  this[visible] = true;
  /** @type {number} */
  this[nodeCount] = 0;
  /** @type {number} */
  this[maxNodeCount] = 200;
  this[strDataUrl] = ;
  this[loadStatus] = LSELoadStatus[LS_UNLOAD];
  /** @type {!Array} */
  this[nodes] = [];
  /** @type {!Array} */
  this[sortNodes] = [];
  this[frustum] = new THREE.Frustum;
  this[viewPort] = new THREE.Vector4;
  this[matLocal] = new THREE.Matrix4;
  this[matLocalInvert] = new THREE.Matrix4;
  this[matModelView] = new THREE.Matrix4;
  this[matVPW] = new THREE.Matrix4;
  this[pixelSizeVector] = new THREE.Vector4;
  this[meshGroup] = new THREE.Group;
  /** @type {number} */
  this[lastAccessFrame] = 0;
  /** @type {number} */
  this[lastAccessTime] = 0;
  /** @type {number} */
  this[maxHttpRequestNum] = 2;
  /** @type {number} */
  this[curHttpRequestNum] = 0;
  /** @type {number} */
  this[maxTexRequestNum] = 2;
  /** @type {number} */
  this[curTexRequestNum] = 0;
  /** @type {number} */
  this[maxNodeParseThreadNum] = 2;
  /** @type {number} */
  this[curNodeParseThreadNum] = 0;
  /** @type {number} */
  this[curLoadingNode] = 0;
  this[bdSphere] = new THREE.Sphere;
};
/** @type {function(): undefined} */
LSJPageLOD[prototype][constructor] = LSJPageLOD;
/**
 * @param {?} local$$49027
 * @return {undefined}
 */
LSJPageLOD[prototype][addNode] = function(local$$49027) {
  this[nodes][push](local$$49027);
  this[sortNodes][push](local$$49027);
  this[meshGroup][add](local$$49027[meshGroup]);
  local$$49027[pageLOD] = this;
  local$$49027[root] = local$$49027;
};
/**
 * @return {?}
 */
LSJPageLOD[prototype][getPixelSizeVector] = function() {
  return this[pixelSizeVector];
};
/**
 * @param {?} local$$49091
 * @return {undefined}
 */
LSJPageLOD[prototype][setPixelSizeVector] = function(local$$49091) {
  this[pixelSizeVector] = local$$49091;
};
/**
 * @return {?}
 */
LSJPageLOD[prototype][getModelViewMatrix] = function() {
  return this[matModelView];
};
/**
 * @return {?}
 */
LSJPageLOD[prototype][getFrustum] = function() {
  return this[frustum];
};
/**
 * @return {?}
 */
LSJPageLOD[prototype][getViewport] = function() {
  return this[viewPort];
};
/**
 * @param {?} local$$49153
 * @return {undefined}
 */
LSJPageLOD[prototype][setViewport] = function(local$$49153) {
  this[viewPort] = local$$49153;
};
/**
 * @param {?} local$$49170
 * @return {undefined}
 */
LSJPageLOD[prototype][setLastAccessTime] = function(local$$49170) {
  this[lastAccessTime] = local$$49170;
};
/**
 * @return {?}
 */
LSJPageLOD[prototype][getLastAccessTime] = function() {
  return this[lastAccessTime];
};
/**
 * @param {?} local$$49202
 * @return {undefined}
 */
LSJPageLOD[prototype][setLastAccessFrame] = function(local$$49202) {
  this[lastAccessFrame] = local$$49202;
};
/**
 * @return {?}
 */
LSJPageLOD[prototype][getLastAccessFrame] = function() {
  return this[lastAccessFrame];
};
/**
 * @param {?} local$$49234
 * @return {undefined}
 */
LSJPageLOD[prototype][addReleaseCount] = function(local$$49234) {
  this[nodeCount] -= local$$49234;
};
/**
 * @param {?} local$$49251
 * @return {undefined}
 */
LSJPageLOD[prototype][addNodeCount] = function(local$$49251) {
  this[nodeCount] += local$$49251;
};
/**
 * @param {?} local$$49268
 * @return {undefined}
 */
LSJPageLOD[prototype][open] = function(local$$49268) {
  if (local$$49268 == ) {
    return;
  }
  var local$$49281 = LSJUtility[createXMLHttp]();
  if (local$$49281 == null) {
    return;
  }
  this[strDataUrl] = local$$49268;
  var local$$49294 = this;
  this[loadStatus] = LSELoadStatus[LS_LOADING];
  /**
   * @return {undefined}
   */
  local$$49281[onreadystatechange] = function() {
    if (local$$49281[readyState] == 4) {
      if (local$$49281[status] == 200) {
        var local$$49321 = local$$49281[responseXML];
        if (!local$$49321 && local$$49281[responseText] != ) {
          local$$49321 = LSJUtility[createXMLDom]();
          if (local$$49321 != null) {
            if (window[ActiveXObject]) {
              local$$49321[loadXML](local$$49281[responseText]);
            } else {
              /** @type {!DOMParser} */
              var local$$49352 = new DOMParser;
              local$$49321 = local$$49352[parseFromString](local$$49281[responseText], text/xml);
            }
          }
        }
        if (local$$49321 != null) {
          var local$$49382 = local$$49321[getElementsByTagName](Scale)[0];
          if (local$$49382 && local$$49382[firstChild]) {
            var local$$49394 = local$$49382[firstChild][nodeValue];
            var local$$49402 = local$$49394[split](,);
            if (local$$49402[length] > 2) {
            }
          }
          local$$49382 = local$$49321[getElementsByTagName](Rotation)[0];
          if (local$$49382 && local$$49382[firstChild]) {
            local$$49394 = local$$49382[firstChild][nodeValue];
            local$$49402 = local$$49394[split](,);
            if (local$$49402[length] > 2) {
            }
          }
          local$$49382 = local$$49321[getElementsByTagName](OffsetMeters)[0];
          if (local$$49382 && local$$49382[firstChild]) {
            local$$49394 = local$$49382[firstChild][nodeValue];
            local$$49402 = local$$49394[split](,);
            if (local$$49402[length] > 2) {
            }
          }
          var local$$49502 = local$$49321[getElementsByTagName](NodeList)[0];
          var local$$49510 = local$$49502[children][length];
          /** @type {number} */
          var local$$49513 = 0;
          for (; local$$49513 < local$$49510; local$$49513++) {
            var local$$49522 = local$$49502[children][local$$49513];
            /** @type {null} */
            local$$49394 = null;
            if (local$$49522[firstChild] != null) {
              local$$49394 = local$$49522[firstChild][nodeValue];
            } else {
              local$$49394 = local$$49522[innerHTML];
            }
            if (local$$49394 != ) {
              var local$$49554 = new LSJPageLODNode;
              local$$49554[strDataPath] = LSJUtility[getAbsolutePath](LSJUtility[getDir](local$$49268), local$$49394);
              local$$49294[addNode](local$$49554);
            }
          }
        }
      }
      this[loadStatus] = LSELoadStatus[LS_LOADED];
    }
  };
  local$$49281[open](GET, local$$49268, true);
  local$$49281[send]();
};
/**
 * @param {?} local$$49620
 * @return {undefined}
 */
LSJPageLOD[prototype][fromJson] = function(local$$49620) {
  if (local$$49620 == ) {
    return;
  }
  this[strDataUrl] = local$$49620;
  var local$$49634 = this;
  this[loadStatus] = LSELoadStatus[LS_LOADING];
  var local$$49646 = new THREE.XHRLoader;
  local$$49646[load](local$$49620, function(local$$49651) {
    var local$$49658 = JSON[parse](local$$49651);
    var local$$49663 = local$$49658[DataDefine];
    if (local$$49663 !== undefined) {
      if (local$$49663[Range] !== undefined) {
        var local$$49687 = new THREE.Vector3(local$$49663[Range].West, local$$49663[Range].South, local$$49663[Range].MinZ);
        var local$$49706 = new THREE.Vector3(local$$49663[Range].East, local$$49663[Range].North, local$$49663[Range].MaxZ);
        var local$$49710 = new THREE.Vector3;
        local$$49710[set](local$$49687[x] / 2 + local$$49706[x] / 2, local$$49687[y] / 2 + local$$49706[y] / 2, local$$49687[z] / 2 + local$$49706[z] / 2);
        var local$$49752 = new THREE.Vector3;
        local$$49752[subVectors](local$$49706, local$$49687);
        local$$49634[bdSphere][set](local$$49710, local$$49752[length]() / 2);
      }
      var local$$49785 = local$$49663[NodeList][Node][length];
      /** @type {number} */
      var local$$49788 = 0;
      for (; local$$49788 < local$$49785; local$$49788++) {
        var local$$49800 = local$$49663[NodeList][Node][local$$49788];
        if (local$$49800 != ) {
          var local$$49806 = new LSJPageLODNode;
          local$$49806[strDataPath] = LSJUtility[getAbsolutePath](LSJUtility[getDir](local$$49620), local$$49800);
          local$$49634[addNode](local$$49806);
        }
      }
      this[loadStatus] = LSELoadStatus[LS_LOADED];
    }
  });
};
/**
 * @param {?} local$$49855
 * @param {?} local$$49856
 * @param {!NodeList} local$$49857
 * @return {undefined}
 */
LSJPageLOD[prototype][addToDropList] = function(local$$49855, local$$49856, local$$49857) {
  if (local$$49856[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
    return;
  }
  /** @type {number} */
  var local$$49872 = 0;
  var local$$49880 = local$$49856[children][length];
  for (; local$$49872 < local$$49880; local$$49872++) {
    this[addToDropList](local$$49855, local$$49856[children][local$$49872], local$$49857);
  }
  if (local$$49856 == local$$49856[root]) {
    return;
  }
  if (local$$49856[strDataPath] == ) {
    return;
  }
  /** @type {number} */
  var local$$49916 = 0;
  var local$$49921 = local$$49857[length];
  for (; local$$49916 < local$$49921; local$$49916++) {
    var local$$49927 = local$$49857[local$$49916];
    var local$$49929;
    var local$$49931;
    local$$49929 = local$$49927[computeNodeLevel]();
    local$$49931 = local$$49856[computeNodeLevel]();
    if (local$$49931 > local$$49929) {
      local$$49857[splice](local$$49916, 0, local$$49856);
      return;
    } else {
      if (local$$49931 == local$$49929) {
        if (local$$49927[parent] == local$$49856[parent]) {
          var local$$49965 = local$$49927[parent];
          if (local$$49965 != null) {
            /** @type {number} */
            var local$$49970 = -1;
            /** @type {number} */
            var local$$49973 = -1;
            /** @type {number} */
            var local$$49976 = 0;
            for (; local$$49976 < local$$49965[children][length]; local$$49976++) {
              if (local$$49965[children][local$$49872] == local$$49927) {
                /** @type {number} */
                local$$49970 = local$$49872;
              }
              if (local$$49965[children][local$$49872] == local$$49856) {
                /** @type {number} */
                local$$49973 = local$$49872;
              }
              if (local$$49970 > -1 && local$$49973 > -1) {
                break;
              }
            }
            if (local$$49973 > local$$49970) {
              local$$49857[splice](local$$49916, 0, local$$49856);
              return;
            }
          }
        }
      }
    }
  }
  local$$49857[push](local$$49856);
};
/**
 * @param {?} local$$50054
 * @param {?} local$$50055
 * @param {?} local$$50056
 * @return {undefined}
 */
LSJPageLOD[prototype][findDropNode] = function(local$$50054, local$$50055, local$$50056) {
  if (local$$50055[getLastAccessFrame]() < this[getLastAccessFrame]()) {
    this[addToDropList](local$$50054, local$$50055, local$$50056);
  } else {
    /** @type {number} */
    var local$$50074 = 0;
    var local$$50082 = local$$50055[children][length];
    for (; local$$50074 < local$$50082; local$$50074++) {
      this[findDropNode](local$$50054, local$$50055[children][local$$50074], local$$50056);
    }
  }
};
/**
 * @param {!Object} local$$50110
 * @param {?} local$$50111
 * @return {?}
 */
LSJPageLOD[prototype][cleanRedundantNodes] = function(local$$50110, local$$50111) {
  if (gdMemUsed < gdMaxMemAllowed) {
    return false;
  }
  if (local$$50110 != null) {
    if (local$$50110[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
      return false;
    }
    /** @type {number} */
    var local$$50137 = 0;
    var local$$50145 = local$$50110[children][length];
    for (; local$$50137 < local$$50145; local$$50137++) {
      this[cleanRedundantNodes](local$$50110[children][local$$50137], local$$50111);
    }
    if (gdMemUsed < gdMaxMemAllowed) {
      return false;
    }
    if (!local$$50111 && local$$50110 == local$$50110[root]) {
      return false;
    }
    if (local$$50110[strDataPath] == ) {
      return false;
    }
    /** @type {number} */
    var local$$50199 = this[getLastAccessFrame]() - local$$50110[getLastAccessFrame]();
    /** @type {number} */
    var local$$50210 = this[getLastAccessTime]() - local$$50110[getLastAccessTime]();
    if (local$$50199 < 1) {
      return false;
    }
    if (local$$50110[isGrandchildrenSafeDel]()) {
      local$$50110[unloadChildren]();
      return true;
    }
    return false;
  }
  return false;
};
/**
 * @param {?} local$$50250
 * @return {undefined}
 */
LSJPageLOD[prototype][computeFrustum] = function(local$$50250) {
  local$$50250[updateMatrixWorld]();
  var local$$50259 = new THREE.Matrix4;
  local$$50259[getInverse](local$$50250[matrixWorld]);
  this[matModelView][multiplyMatrices](local$$50259, this[matLocal]);
  this[matVPW][multiplyMatrices](local$$50250[projectionMatrix], this[matModelView]);
  this[frustum][setFromMatrix](this[matVPW]);
  this[pixelSizeVector] = LSJMath[computePixelSizeVector](this[viewPort], local$$50250[projectionMatrix], this[matModelView]);
};
/**
 * @param {?} local$$50328
 * @param {?} local$$50329
 * @return {?}
 */
function nodeDistCompare(local$$50328, local$$50329) {
  return local$$50328[distToEyeSquare] - local$$50329[distToEyeSquare];
}
/**
 * @param {?} local$$50347
 * @return {undefined}
 */
LSJPageLOD[prototype][update] = function(local$$50347) {
  var local$$50352 = this[nodes];
  this[lastAccessTime] = (new Date)[getTime]();
  ++this[lastAccessFrame];
  this[computeFrustum](local$$50347);
  /** @type {number} */
  this[curLoadingNode] = 0;
  /** @type {number} */
  var local$$50381 = 0;
  var local$$50386 = local$$50352[length];
  for (; local$$50381 < local$$50386; local$$50381++) {
    local$$50352[local$$50381][update](local$$50347);
  }
  if (gdMemUsed > gdMaxMemAllowed) {
    /** @type {number} */
    var local$$50402 = 0;
    var local$$50407 = local$$50352[length];
    for (; local$$50402 < local$$50407; local$$50402++) {
      this[cleanRedundantNodes](this[nodes][local$$50402], false);
    }
  }
};
/**
 * @return {undefined}
 */
LSJNodeMaterial = function() {
  /** @type {number} */
  this[id] = -1;
  this[status] = LSELoadStatus[LS_UNLOAD];
  this[imgUrl] = ;
  /** @type {null} */
  this[material] = null;
  /** @type {boolean} */
  this[bImgBlobUrl] = false;
};
/**
 * @return {undefined}
 */
LSJPageLODNode = function() {
  this[type] = PageLODNode;
  /** @type {!Array} */
  this[children] = [];
  /** @type {!Array} */
  this[childRanges] = [];
  /** @type {null} */
  this[pageLOD] = null;
  /** @type {null} */
  this[parent] = null;
  /** @type {null} */
  this[root] = null;
  this[strDataPath] = ;
  this[meshGroup] = new THREE.Group;
  /** @type {boolean} */
  this[bNormalRendered] = false;
  /** @type {boolean} */
  this[bInFrustumTestOk] = false;
  this[bdSphere] = new THREE.Sphere;
  this[bdBox] = new THREE.Box3;
  this[btLoadStatus] = LSELoadStatus[LS_UNLOAD];
  /** @type {number} */
  this[enRangeMode] = 0;
  /** @type {number} */
  this[lastAccessFrame] = 0;
  /** @type {number} */
  this[lastAccessTime] = 0;
  /** @type {boolean} */
  this[bHasGeometry] = false;
  /** @type {!Array} */
  this[arryMaterials] = [];
  /** @type {!Array} */
  this[arryMaterialUsed] = [];
  /** @type {null} */
  this[dataBuffer] = null;
  /** @type {number} */
  this[distToEyeSquare] = 0;
  /** @type {number} */
  this[dMemUsed] = 0;
};
/** @type {function(): undefined} */
LSJPageLODNode[prototype][constructor] = LSJPageLODNode;
/**
 * @param {?} local$$50629
 * @return {undefined}
 */
LSJPageLODNode[prototype][setInFrustumTestOk] = function(local$$50629) {
  this[bInFrustumTestOk] = local$$50629;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][isInFrustumTestOk] = function() {
  return this[bInFrustumTestOk];
};
/**
 * @param {?} local$$50661
 * @return {undefined}
 */
LSJPageLODNode[prototype][setLoadStatus] = function(local$$50661) {
  this[btLoadStatus] = local$$50661;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][hasGeometry] = function() {
  return this[bHasGeometry];
};
/**
 * @param {?} local$$50693
 * @return {undefined}
 */
LSJPageLODNode[prototype][setHasGeometry] = function(local$$50693) {
  this[bHasGeometry] = local$$50693;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][getLoadStatus] = function() {
  return this[btLoadStatus];
};
/**
 * @param {?} local$$50725
 * @return {undefined}
 */
LSJPageLODNode[prototype][setLastAccessTime] = function(local$$50725) {
  this[lastAccessTime] = local$$50725;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][getLastAccessTime] = function() {
  return this[lastAccessTime];
};
/**
 * @param {?} local$$50757
 * @return {undefined}
 */
LSJPageLODNode[prototype][setLastAccessFrame] = function(local$$50757) {
  this[lastAccessFrame] = local$$50757;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][getLastAccessFrame] = function() {
  return this[lastAccessFrame];
};
/**
 * @param {?} local$$50789
 * @return {undefined}
 */
LSJPageLODNode[prototype][addNode] = function(local$$50789) {
  this[children][push](local$$50789);
  local$$50789[pageLOD] = this[pageLOD];
  local$$50789[root] = this[root];
  local$$50789[parent] = this;
  this[meshGroup][add](local$$50789[meshGroup]);
};
/**
 * @param {?} local$$50841
 * @param {?} local$$50842
 * @param {?} local$$50843
 * @return {?}
 */
LSJPageLODNode[prototype][loadTexture] = function(local$$50841, local$$50842, local$$50843) {
  if (local$$50842[curTexRequestNum] > local$$50842[maxTexRequestNum]) {
    return;
  }
  local$$50841[status] = LSELoadStatus[LS_LOADING];
  local$$50842[curTexRequestNum]++;
  var local$$50871 = new THREE.Texture;
  var local$$50879 = document[createElement](img);
  local$$50879[src] = local$$50841[imgUrl];
  var local$$50889;
  /**
   * @param {?} local$$50894
   * @param {?} local$$50895
   * @return {undefined}
   */
  local$$50879[onerror] = function(local$$50894, local$$50895) {
    if (local$$50841[bImgBlobUrl]) {
      window[URL][revokeObjectURL](local$$50841[imgUrl]);
    }
    local$$50841[status] = LSELoadStatus[LS_LOADED];
    local$$50842[curTexRequestNum]--;
  };
  /**
   * @param {?} local$$50934
   * @param {?} local$$50935
   * @return {undefined}
   */
  local$$50879[onload] = function(local$$50934, local$$50935) {
    window[URL][revokeObjectURL](local$$50879[src]);
    local$$50871[image] = local$$50879;
    /** @type {boolean} */
    local$$50871[needsUpdate] = true;
    local$$50871[side] = THREE[FrontSide];
    local$$50871[wrapS] = THREE[RepeatWrapping];
    local$$50871[wrapT] = THREE[RepeatWrapping];
    local$$50871[minFilter] = THREE[LinearFilter];
    local$$50871[magFilter] = THREE[LinearFilter];
    /** @type {boolean} */
    local$$50871[generateMipmaps] = false;
    /** @type {number} */
    var local$$51006 = 3;
    if (local$$50871[format] == THREE[RGBAFormat]) {
      /** @type {number} */
      local$$51006 = 4;
    }
    /** @type {number} */
    var local$$51029 = local$$50879[width] * local$$50879[height] * local$$51006;
    gdMemUsed = gdMemUsed + local$$51029;
    local$$50841[material][map] = local$$50871;
    /** @type {boolean} */
    local$$50841[material][needsUpdate] = true;
    if (local$$50841[bImgBlobUrl]) {
      window[URL][revokeObjectURL](local$$50841[imgUrl]);
    }
    local$$50841[imgUrl] = ;
    local$$50841[status] = LSELoadStatus[LS_LOADED];
    local$$50842[curTexRequestNum]--;
  };
  return local$$50871;
};
/**
 * @return {undefined}
 */
LSJPageLODNode[prototype][netLoad] = function() {
  if (this[pageLOD][curHttpRequestNum] > this[pageLOD][maxHttpRequestNum]) {
    return;
  }
  /**
   * @param {?} local$$51121
   * @return {undefined}
   */
  var local$$51125 = function(local$$51121) {
  };
  /**
   * @param {?} local$$51127
   * @return {undefined}
   */
  var local$$51146 = function(local$$51127) {
    local$$51129[setLoadStatus](LSELoadStatus.LS_NET_LOADED);
    local$$51129[pageLOD][curHttpRequestNum]--;
  };
  this[setLoadStatus](LSELoadStatus.LS_NET_LOADING);
  /** @type {!XMLHttpRequest} */
  var local$$51155 = new XMLHttpRequest;
  local$$51155[open](GET, this[strDataPath], true);
  local$$51155[responseType] = arraybuffer;
  this[pageLOD][curHttpRequestNum]++;
  local$$51155[send](null);
  var local$$51129 = this;
  /**
   * @return {undefined}
   */
  local$$51155[onreadystatechange] = function() {
    if (local$$51155[readyState] == 4) {
      if (local$$51155[status] == 200) {
        local$$51129[dataBuffer] = local$$51155[response];
      } else {
      }
      local$$51129[setLoadStatus](LSELoadStatus.LS_NET_LOADED);
      local$$51129[pageLOD][curHttpRequestNum]--;
    }
  };
};
/**
 * @return {undefined}
 */
LSJPageLODNode[prototype][load] = function() {
  if (this[pageLOD][curNodeParseThreadNum] > this[pageLOD][maxNodeParseThreadNum]) {
    return;
  }
  if (this[dataBuffer] == null) {
    this[setLoadStatus](LSELoadStatus.LS_LOADED);
    return;
  }
  this[setLoadStatus](LSELoadStatus.LS_LOADING);
  var local$$51285 = this;
  this[pageLOD][curNodeParseThreadNum]++;
  /** @type {!Worker} */
  var local$$51299 = new Worker(script/lsjworker/LSJPWM.min.js);
  /**
   * @param {?} local$$51304
   * @return {undefined}
   */
  local$$51299[onmessage] = function(local$$51304) {
    var local$$51309 = local$$51304[data];
    if (local$$51309 != null && local$$51309 != undefined) {
      var local$$51315;
      {
        var local$$51320 = local$$51285[strDataPath];
        local$$51315 = local$$51320[substr](0, local$$51320[lastIndexOf](/) + 1);
      }
      var local$$51344 = local$$51309[arryMaterials][length];
      /** @type {number} */
      var local$$51347 = 0;
      for (; local$$51347 < local$$51344; local$$51347++) {
        var local$$51356 = local$$51309[arryMaterials][local$$51347];
        var local$$51359 = new LSJNodeMaterial;
        if (local$$51356[imgUrl] != ) {
          if (local$$51356[bUrl]) {
            local$$51359[imgUrl] = local$$51315 + local$$51356[imgUrl];
          } else {
            local$$51359[imgUrl] = local$$51356[imgUrl];
          }
        } else {
          if (local$$51356[imgBlob] != null) {
            local$$51359[imgUrl] = window[URL][createObjectURL](local$$51356[imgBlob]);
            /** @type {null} */
            local$$51356[imgBlob] = null;
            /** @type {boolean} */
            local$$51359[bImgBlobUrl] = true;
          }
        }
        local$$51359[material] = new THREE.MeshBasicMaterial;
        local$$51359[material][color] = (new THREE.Color)[setRGB](local$$51356[diffuseR], local$$51356[diffuseG], local$$51356[diffuseB]);
        if (local$$51359[imgUrl] == ) {
          local$$51359[status] = LSELoadStatus[LS_LOADED];
        }
        local$$51285[arryMaterials][push](local$$51359);
      }
      local$$51285[parse](local$$51309, local$$51285[arryMaterials], local$$51315);
    }
    /** @type {null} */
    local$$51309 = null;
    /** @type {null} */
    local$$51304[data] = null;
    /** @type {null} */
    local$$51285[dataBuffer] = null;
    local$$51285[setLoadStatus](LSELoadStatus.LS_LOADED);
    local$$51285[pageLOD][curNodeParseThreadNum]--;
  };
  /**
   * @param {?} local$$51533
   * @return {undefined}
   */
  local$$51299[onerror] = function(local$$51533) {
    console[log](Error: + local$$51533[message]);
    /** @type {null} */
    local$$51285[dataBuffer] = null;
    local$$51285[setLoadStatus](LSELoadStatus.LS_LOADED);
    local$$51285[pageLOD][curNodeParseThreadNum]--;
  };
  local$$51299[postMessage](this[dataBuffer]);
};
/**
 * @param {!Array} local$$51588
 * @param {?} local$$51589
 * @param {?} local$$51590
 * @return {undefined}
 */
LSJPageLODNode[prototype][parse] = function(local$$51588, local$$51589, local$$51590) {
  if (local$$51588 == null || local$$51588 === undefined) {
    return;
  }
  /** @type {number} */
  var local$$51601 = 0;
  var local$$51609 = local$$51588[children][length];
  /** @type {number} */
  local$$51601 = 0;
  for (; local$$51601 < local$$51609; local$$51601++) {
    var local$$51618 = new LSJPageLODNode;
    this[addNode](local$$51618);
    local$$51618[parse](local$$51588[children][local$$51601], local$$51589, local$$51590);
  }
  this[enRangeMode] = local$$51588[enRangeMode];
  if (local$$51588[childRanges][length] > 0) {
    /** @type {number} */
    local$$51609 = local$$51588[childRanges][length] / 2;
    /** @type {number} */
    local$$51601 = 0;
    for (; local$$51601 < local$$51609; local$$51601++) {
      var local$$51671 = new THREE.Vector2;
      local$$51671[x] = local$$51588[childRanges][2 * local$$51601];
      local$$51671[y] = local$$51588[childRanges][2 * local$$51601 + 1];
      this[childRanges][push](local$$51671);
    }
  }
  if (this[strDataPath] == ) {
    if (local$$51588[strDataPath] != ) {
      this[strDataPath] = local$$51590 + local$$51588[strDataPath];
    }
  }
  if (local$$51588[bdSphere][length] > 0) {
    this[bdSphere] = new THREE.Sphere;
    var local$$51753 = new THREE.Vector3;
    local$$51753[set](local$$51588[bdSphere][0], local$$51588[bdSphere][1], local$$51588[bdSphere][2]);
    this[bdSphere][set](local$$51753, local$$51588[bdSphere][3]);
    LSJMath[expandSphere](this[pageLOD][bdSphere], this[bdSphere]);
  }
  /** @type {number} */
  this[dMemUsed] = 0;
  var local$$51817 = local$$51588[nodeMeshes][length];
  /** @type {number} */
  var local$$51820 = 0;
  for (; local$$51820 < local$$51817; local$$51820++) {
    var local$$51829 = local$$51588[nodeMeshes][local$$51820];
    if (local$$51829[verts] != null && local$$51829[matIndex] >= 0 && local$$51829[matIndex] < local$$51589[length]) {
      var local$$51852 = new THREE.BufferGeometry;
      if (local$$51829[indices] != null) {
        local$$51852[setIndex](new THREE.BufferAttribute(local$$51829[indices], 1));
        this[dMemUsed] += local$$51829[indices][length] * 2;
      }
      if (local$$51829[verts] != null) {
        local$$51852[addAttribute](position, new THREE.BufferAttribute(local$$51829[verts], 3));
        this[dMemUsed] += local$$51829[verts][length] * 4;
      }
      if (local$$51829[normals] != null) {
        local$$51852[addAttribute](normal, new THREE.BufferAttribute(local$$51829[normals], 3));
        this[dMemUsed] += local$$51829[normals][length] * 4;
      }
      if (local$$51829[colors] != null) {
        local$$51852[addAttribute](color, new THREE.BufferAttribute(local$$51829[colors], local$$51829[colorPerNum]));
        this[dMemUsed] += local$$51829[colors][length] * 4;
      }
      var local$$51996 = local$$51829[uvs][length];
      /** @type {number} */
      k = 0;
      for (; k < local$$51996; k++) {
        if (local$$51829[uvs][k] != null && local$$51829[uvs][k] != undefined) {
          local$$51852[addAttribute](uv, new THREE.BufferAttribute(local$$51829[uvs][k], 2));
          this[dMemUsed] += local$$51829[uvs][k][length] * 4;
        }
      }
      var local$$52054 = local$$51589[local$$51829[matIndex]];
      var local$$52061 = new THREE.Mesh(local$$51852, local$$52054[material]);
      this[arryMaterialUsed][push](local$$52054);
      this[meshGroup][add](local$$52061);
      this[setHasGeometry](true);
      this[pageLOD][addNodeCount](1);
    }
  }
  gdMemUsed = gdMemUsed + this[dMemUsed];
  if (this[strDataPath] == ) {
    this[btLoadStatus] = LSELoadStatus[LS_LOADED];
  }
};
/**
 * @param {?} local$$52131
 * @return {?}
 */
LSJPageLODNode[prototype][checkInFrustum] = function(local$$52131) {
  this[setInFrustumTestOk](false);
  var local$$52146 = this[pageLOD][getFrustum]();
  if (!this[bdSphere][empty]()) {
    if (!local$$52146[intersectsSphere](this[bdSphere])) {
      return false;
    }
  } else {
    if (!this[bdBox][empty]()) {
      if (!local$$52146[intersectsBox](this[bdBox])) {
        return false;
      }
    }
  }
  this[setInFrustumTestOk](true);
  return true;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][computeNodeLevel] = function() {
  /** @type {number} */
  var local$$52214 = 0;
  var local$$52219 = this[parent];
  for (; local$$52219 != null;) {
    local$$52214++;
    local$$52219 = local$$52219[parent];
  }
  return local$$52214;
};
/**
 * @param {?} local$$52246
 * @return {?}
 */
LSJPageLODNode[prototype][computeDistSquare2Eye] = function(local$$52246) {
  /** @type {number} */
  var local$$52249 = 0;
  if (!this[bdSphere][empty]()) {
    local$$52249 = LSJMath[computeTowVecDistSquare](this[bdSphere][center], local$$52246[position]);
    var local$$52291 = LSJMath[computeDistFromEye](this[bdSphere][center], this[pageLOD][getModelViewMatrix]());
    /** @type {number} */
    var local$$52294 = local$$52291 * local$$52291;
    return local$$52249;
  } else {
    if (!this[bdBox][empty]()) {
      local$$52294 = LSJMath[computeDistFromEye](this[bdBox][center], this[pageLOD][getModelViewMatrix]());
      local$$52249 = LSJMath[computeTowVecDistSquare](this[bdBox][center], local$$52246[position]);
      return local$$52249;
    }
  }
  /** @type {number} */
  var local$$52347 = 0;
  var local$$52355 = this[children][length];
  for (; local$$52347 < local$$52355; local$$52347++) {
    var local$$52364 = this[children][local$$52347];
    if (local$$52364 != null) {
      local$$52249 = local$$52364[computeDistSquare2Eye](local$$52246);
      if (local$$52249 > -1) {
        return local$$52249;
      }
    }
  }
  return -2;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][isGrandchildrenSafeDel] = function() {
  if (this[getLoadStatus]() != LSELoadStatus[LS_UNLOAD] && this[getLoadStatus]() != LSELoadStatus[LS_NET_LOADED] && this[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
    return false;
  }
  if (this[hasLoadingMaterial]()) {
    return false;
  }
  /** @type {number} */
  var local$$52438 = 0;
  var local$$52446 = this[children][length];
  for (; local$$52438 < local$$52446; local$$52438++) {
    if (!this[children][local$$52438][isGrandchildrenSafeDel]()) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][isAllMaterialLoaded] = function() {
  /** @type {number} */
  var local$$52481 = 0;
  var local$$52489 = this[arryMaterialUsed][length];
  for (; local$$52481 < local$$52489; local$$52481++) {
    if (this[arryMaterialUsed][local$$52481][status] != LSELoadStatus[LS_LOADED]) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][hasLoadingMaterial] = function() {
  /** @type {number} */
  var local$$52526 = 0;
  var local$$52534 = this[arryMaterialUsed][length];
  for (; local$$52526 < local$$52534; local$$52526++) {
    if (this[arryMaterialUsed][local$$52526][status] != LSELoadStatus[LS_UNLOAD] && this[arryMaterialUsed][local$$52526][status] != LSELoadStatus[LS_LOADED]) {
      return true;
    }
  }
  return false;
};
/**
 * @return {?}
 */
LSJPageLODNode[prototype][calcNodeCount] = function() {
  /** @type {number} */
  var local$$52583 = 0;
  if (this[hasGeometry]()) {
    /** @type {number} */
    local$$52583 = local$$52583 + 1;
  }
  /** @type {number} */
  var local$$52597 = 0;
  var local$$52605 = this[children][length];
  for (; local$$52597 < local$$52605; local$$52597++) {
    local$$52583 = local$$52583 + this[children][local$$52597][calcNodeCount]();
  }
  return local$$52583;
};
/**
 * @return {undefined}
 */
LSJPageLODNode[prototype][unloadChildren] = function() {
  /** @type {number} */
  var local$$52637 = 0;
  /** @type {number} */
  var local$$52640 = 0;
  var local$$52648 = this[children][length];
  /** @type {number} */
  local$$52640 = 0;
  for (; local$$52640 < local$$52648; local$$52640++) {
    this[children][local$$52640][unloadChildren]();
  }
  this[children][splice](0, local$$52648);
  this[childRanges][splice](0, this[childRanges][length]);
  this[arryMaterialUsed][splice](0, this[arryMaterialUsed][length]);
  for (; this[arryMaterials][length] > 0;) {
    var local$$52724 = this[arryMaterials][pop]();
    if (local$$52724[material] != null && local$$52724[material] != undefined) {
      var local$$52742 = local$$52724[material][map];
      if (local$$52742 != null && local$$52742 != undefined) {
        if (local$$52742[image] != null) {
          /** @type {number} */
          var local$$52754 = 3;
          if (local$$52742[format] == THREE[RGBAFormat]) {
            /** @type {number} */
            local$$52754 = 4;
          }
          /** @type {number} */
          var local$$52783 = local$$52742[image][width] * local$$52742[image][height] * local$$52754;
          /** @type {number} */
          gdMemUsed = gdMemUsed - local$$52783;
          /** @type {null} */
          local$$52742[image] = null;
        }
        local$$52742[dispose]();
      }
      local$$52724[material][dispose]();
      /** @type {null} */
      local$$52724[material][map] = null;
      /** @type {null} */
      local$$52724[material] = null;
    }
  }
  /** @type {number} */
  var local$$52844 = this[meshGroup][children][length] - 1;
  for (; local$$52844 >= 0; local$$52844--) {
    var local$$52857 = this[meshGroup][children][local$$52844];
    this[meshGroup][remove](local$$52857);
    if (local$$52857 != null && local$$52857 instanceof THREE[Mesh]) {
      if (local$$52857[geometry]) {
        local$$52857[geometry][dispose]();
      }
      if (local$$52857[material] != null && local$$52857[material] != undefined) {
        if (local$$52857[material][map] != null && local$$52857[material][map] != undefined) {
          local$$52857[material][map][dispose]();
        }
        local$$52857[material][dispose]();
      }
      /** @type {null} */
      local$$52857[material] = null;
      /** @type {null} */
      local$$52857[geometry] = null;
      this[pageLOD][addReleaseCount](1);
    }
    /** @type {null} */
    local$$52857 = null;
  }
  /** @type {number} */
  gdMemUsed = gdMemUsed - this[dMemUsed];
  /** @type {number} */
  this[dMemUsed] = 0;
  /** @type {boolean} */
  this[bHasGeometry] = false;
  /** @type {null} */
  this[dataBuffer] = null;
  this[setLoadStatus](LSELoadStatus.LS_UNLOAD);
};
/**
 * @param {?} local$$53009
 * @return {?}
 */
LSJPageLODNode[prototype][checkAllGroupChildLoaded] = function(local$$53009) {
  /** @type {number} */
  var local$$53012 = 0;
  var local$$53020 = this[children][length];
  for (; local$$53012 < local$$53020; local$$53012++) {
    var local$$53029 = this[children][local$$53012];
    if (local$$53029 != null) {
      if (local$$53029[checkInFrustum](local$$53009) && local$$53029[children][length] > 1) {
        local$$53029[setInFrustumTestOk](true);
        var local$$53057 = local$$53029[children][0];
        if (local$$53057) {
          if (local$$53057[strDataPath] != ) {
            if (local$$53057[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
};
/**
 * @param {?} local$$53100
 * @return {?}
 */
LSJPageLODNode[prototype][update] = function(local$$53100) {
  /** @type {boolean} */
  this[meshGroup][visible] = false;
  var local$$53120 = this[meshGroup][children][length];
  /** @type {number} */
  var local$$53123 = 0;
  /** @type {number} */
  local$$53123 = 0;
  for (; local$$53123 < local$$53120; local$$53123++) {
    /** @type {boolean} */
    this[meshGroup][children][local$$53123][visible] = false;
  }
  /** @type {boolean} */
  this[bNormalRendered] = false;
  if (!this[checkInFrustum](local$$53100)) {
    /** @type {boolean} */
    this[meshGroup][visible] = false;
    return false;
  }
  this[setLastAccessTime](this[pageLOD][getLastAccessTime]());
  this[setLastAccessFrame](this[pageLOD][getLastAccessFrame]());
  if (this[strDataPath] != ) {
    if (this[getLoadStatus]() == LSELoadStatus[LS_UNLOAD]) {
      this[netLoad]();
    }
    if (this[getLoadStatus]() == LSELoadStatus[LS_NET_LOADED]) {
      this[load]();
    }
    if (this[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
      this[pageLOD][curLoadingNode]++;
      return false;
    }
  }
  /** @type {number} */
  var local$$53258 = 0;
  if (this[childRanges][length] > 0) {
    if (this[enRangeMode] == LSERangeMode[RM_DISTANCE_FROM_EYE_POINT]) {
      if (!this[bdSphere][empty]()) {
        local$$53258 = LSJMath[computeDistFromEye](this[bdSphere][center], this[pageLOD][getModelViewMatrix]());
      }
    } else {
      if (this[enRangeMode] == LSERangeMode[RM_PIXEL_SIZE_ON_SCREEN]) {
        if (!this[bdSphere][empty]()) {
          /** @type {number} */
          local$$53258 = LSJMath[computeSpherePixelSize](this[bdSphere], this[pageLOD][getPixelSizeVector]()) * .5;
        }
      }
    }
  }
  /** @type {boolean} */
  var local$$53348 = true;
  /** @type {number} */
  var local$$53351 = 0;
  local$$53120 = this[children][length];
  if (this[childRanges][length] > 0) {
    /** @type {number} */
    local$$53123 = 0;
    for (; local$$53123 < local$$53120; local$$53123++) {
      var local$$53379 = this[children][local$$53123];
      if (local$$53123 < this[childRanges][length]) {
        var local$$53392 = this[childRanges][local$$53123];
        if (local$$53379 && local$$53258 >= local$$53392[x] && local$$53258 < local$$53392[y]) {
          if (local$$53379[update](local$$53100)) {
            /** @type {boolean} */
            this[bNormalRendered] = true;
          }
        }
      } else {
        if (local$$53379 && local$$53379[update](local$$53100)) {
          /** @type {boolean} */
          this[bNormalRendered] = true;
        }
      }
    }
    if (!this[bNormalRendered] && local$$53120 > 0) {
      local$$53379 = this[children][0];
      if (local$$53379 && local$$53379[update](local$$53100)) {
        /** @type {boolean} */
        this[bNormalRendered] = true;
      }
    }
  } else {
    /** @type {number} */
    local$$53123 = 0;
    for (; local$$53123 < local$$53120; local$$53123++) {
      local$$53379 = this[children][local$$53123];
      if (local$$53379 && local$$53379[update](local$$53100)) {
        /** @type {boolean} */
        this[bNormalRendered] = true;
      }
    }
  }
  /** @type {boolean} */
  this[meshGroup][visible] = true;
  /** @type {boolean} */
  var local$$53507 = false;
  var local$$53513 = this[isAllMaterialLoaded]();
  if (!this[bNormalRendered] && this[pageLOD][curTexRequestNum] < this[pageLOD][maxTexRequestNum]) {
    local$$53120 = this[arryMaterialUsed][length];
    /** @type {number} */
    local$$53123 = 0;
    for (; local$$53123 < local$$53120; local$$53123++) {
      var local$$53551 = this[arryMaterialUsed][local$$53123];
      if (local$$53551[status] == LSELoadStatus[LS_UNLOAD]) {
        this[loadTexture](local$$53551, this[pageLOD]);
      }
    }
  }
  local$$53120 = this[meshGroup][children][length];
  /** @type {number} */
  local$$53123 = 0;
  for (; local$$53123 < local$$53120; local$$53123++) {
    var local$$53599 = this[meshGroup][children][local$$53123];
    if (local$$53599 && local$$53599 instanceof THREE[Mesh]) {
      if (this[bdSphere][radius] <= 0 && local$$53599[geometry][boundingSphere] != null) {
        LSJMath[expandSphere](this[pageLOD][bdSphere], local$$53599[geometry][boundingSphere]);
      }
      if (local$$53513) {
        /** @type {boolean} */
        local$$53599[visible] = true;
        /** @type {boolean} */
        local$$53599[material][wireframe] = displayMode === DisplayMode[Wireframe];
        /** @type {boolean} */
        local$$53507 = true;
      } else {
        /** @type {boolean} */
        local$$53599[visible] = false;
      }
    }
  }
  if (!local$$53507) {
    this[meshGroup][visible] = this[bNormalRendered];
    return this[bNormalRendered];
  }
  /** @type {boolean} */
  this[bNormalRendered] = true;
  return true;
};
/**
 * @param {(Object|string)} local$$53714
 * @param {?} local$$53715
 * @param {!Object} local$$53716
 * @return {undefined}
 */
EditorLineControls = function(local$$53714, local$$53715, local$$53716) {
  local$$53716 = local$$53716 !== undefined ? local$$53716 : document;
  /** @type {boolean} */
  this[enabled] = true;
  this[object] = undefined;
  this[scene] = local$$53715;
  /** @type {!Array} */
  this[splineHelperObjects] = [];
  /** @type {number} */
  this[splinePointsLength] = 0;
  this[geometry] = new THREE.SphereGeometry(5, 50, 50);
  /** @type {number} */
  this[ARC_SEGMENTS] = 200;
  var local$$53769 = NodeEdit;
  var local$$53771 = this;
  this[transformControl] = new THREE.TransformControls(local$$53714, local$$53716);
  local$$53715[add](this[transformControl]);
  var local$$53788;
  /**
   * @return {undefined}
   */
  local$$53771[delayHideTransform] = function() {
    local$$53771[cancelHideTransorm]();
    local$$53771[hideTransform]();
  };
  /**
   * @return {undefined}
   */
  local$$53771[hideTransform] = function() {
    /** @type {number} */
    local$$53788 = setTimeout(function() {
      local$$53771[transformControl][detach](local$$53771[transformControl][object]);
    }, 2500);
  };
  /**
   * @return {undefined}
   */
  local$$53771[cancelHideTransorm] = function() {
    if (local$$53771[hiding]) {
      local$$53771[clearTimeout](local$$53771[hiding]);
    }
  };
  this[transformControl][addEventListener](change, function(local$$53866) {
    local$$53771[cancelHideTransorm]();
  });
  this[transformControl][addEventListener](mouseDown, function(local$$53885) {
    local$$53771[cancelHideTransorm]();
  });
  this[transformControl][addEventListener](mouseUp, function(local$$53904) {
    local$$53771[delayHideTransform]();
  });
  this[transformControl][addEventListener](objectChange, function(local$$53923) {
    /** @type {!Array} */
    local$$53771[object][points] = [];
    /** @type {number} */
    i = 0;
    for (; i < local$$53771[splinePointsLength]; i++) {
      local$$53771[object][points][push](local$$53771[splineHelperObjects][i][position]);
    }
    local$$53771[updateSplineOutline]();
  });
  this[dragcontrols] = new THREE.DragControls(local$$53714, this[splineHelperObjects], local$$53716);
  this[dragcontrols][on](hoveron, function(local$$53992) {
    local$$53771[transformControl][attach](local$$53992[object]);
    local$$53771[cancelHideTransorm]();
  });
  this[dragcontrols][on](hoveroff, function(local$$54022) {
    if (local$$54022) {
      local$$53771[delayHideTransform]();
    }
  });
  /**
   * @param {?} local$$54038
   * @return {undefined}
   */
  this[addSplineObject] = function(local$$54038) {
    var local$$54055 = new THREE.Mesh(this[geometry], new THREE.MeshLambertMaterial({
      color : Math[random]() * 16777215
    }));
    local$$54055[material][ambient] = local$$54055[material][color];
    if (local$$54038) {
      local$$54055[position][copy](local$$54038);
    }
    /** @type {boolean} */
    local$$54055[castShadow] = true;
    /** @type {boolean} */
    local$$54055[receiveShadow] = true;
    this[scene][add](local$$54055);
    this[splineHelperObjects][push](local$$54055);
  };
  /**
   * @param {?} local$$54117
   * @return {undefined}
   */
  this[addPoint] = function(local$$54117) {
    if (!this[enabled]) {
      return;
    }
    this[splinePointsLength]++;
    local$$53771[object][points][push](local$$54117);
    this[addSplineObject](local$$54117);
    this[updateSplineOutline]();
  };
  /**
   * @return {undefined}
   */
  this[removePoint] = function() {
    if (this[splinePointsLength] <= 0 || !this[enabled]) {
      return;
    }
    this[splinePointsLength]--;
    local$$53771[object][points][pop]();
    this[scene][remove](this[splineHelperObjects][pop]());
    this[updateSplineOutline]();
  };
  /**
   * @return {undefined}
   */
  this[updateSplineOutline] = function() {
    if (this[splinePointsLength] < 2) {
      return;
    }
    var local$$54228;
    var local$$54236 = this[object][mesh];
    /** @type {number} */
    var local$$54239 = 0;
    for (; local$$54239 < this[ARC_SEGMENTS]; local$$54239++) {
      local$$54228 = local$$54236[geometry][vertices][local$$54239];
      local$$54228[copy](this[object][getPoint](local$$54239 / (this[ARC_SEGMENTS] - 1)));
    }
    /** @type {boolean} */
    local$$54236[geometry][verticesNeedUpdate] = true;
  };
  /**
   * @param {?} local$$54293
   * @return {undefined}
   */
  this[setMode] = function(local$$54293) {
    local$$53769 = local$$54293 ? local$$54293 : local$$53769;
    /** @type {boolean} */
    var local$$54304 = local$$53769 == NodeEdit ? true : false;
    /** @type {number} */
    i = 0;
    for (; i < local$$53771[splinePointsLength]; i++) {
      /** @type {boolean} */
      local$$53771[splineHelperObjects][i][visible] = local$$54304;
    }
  };
  /**
   * @return {undefined}
   */
  this[update] = function() {
    this[transformControl][update]();
  };
  /**
   * @param {?} local$$54349
   * @return {undefined}
   */
  this[attach] = function(local$$54349) {
    this[object] = local$$54349;
    /** @type {boolean} */
    this[enabled] = true;
    this[splinePointsLength] = local$$53771[object][points][length];
    /** @type {number} */
    var local$$54377 = 0;
    for (; local$$54377 < this[splinePointsLength]; local$$54377++) {
      this[addSplineObject](local$$53771[object][points][local$$54377]);
    }
  };
  /**
   * @return {undefined}
   */
  this[detach] = function() {
    this[object] = undefined;
    /** @type {boolean} */
    this[enabled] = false;
  };
};
/**
 * @param {?} local$$54429
 * @return {undefined}
 */
LSJFlyAroundCenterControls = function(local$$54429) {
  this[camera] = local$$54429;
  this[target] = new THREE.Vector3(0, 0, 0);
  /** @type {boolean} */
  this[enable] = false;
  /** @type {number} */
  this[delta] = .1;
  /** @type {number} */
  this[theta] = 0;
  var local$$54464 = this;
  /**
   * @param {?} local$$54469
   * @param {?} local$$54470
   * @return {undefined}
   */
  this[FlyAroundCenter] = function(local$$54469, local$$54470) {
    this[target] = local$$54469;
    /** @type {boolean} */
    this[enable] = true;
    this[delta] = local$$54470;
    var local$$54490 = new THREE.Vector3;
    local$$54490[copy](this[camera][position]);
    local$$54490[sub](this[target]);
    /** @type {number} */
    local$$54490[z] = 0;
    var local$$54521 = local$$54490[length]();
    if (local$$54521 == 0) {
      /** @type {number} */
      this[theta] = 0;
      return;
    }
    if (local$$54490[x] > 0) {
      if (local$$54490[y] > 0) {
        this[theta] = THREE[Math][radToDeg](Math[asin](Math[abs](local$$54490[x]) / local$$54521));
      } else {
        /** @type {number} */
        this[theta] = 180 - THREE[Math][radToDeg](Math[asin](Math[abs](local$$54490[x]) / local$$54521));
      }
    } else {
      if (local$$54490[y] > 0) {
        /** @type {number} */
        this[theta] = -THREE[Math][radToDeg](Math[asin](Math[abs](local$$54490[x]) / local$$54521));
      } else {
        this[theta] = 180 + THREE[Math][radToDeg](Math[asin](Math[abs](local$$54490[x]) / local$$54521));
      }
    }
  };
  /**
   * @return {undefined}
   */
  this[stop] = function() {
    /** @type {boolean} */
    this[enable] = false;
    this[target] = undefined;
    /** @type {boolean} */
    this[enable] = false;
    /** @type {number} */
    this[delta] = .1;
    /** @type {number} */
    this[theta] = 0;
  };
  /**
   * @return {undefined}
   */
  this[update] = function() {
    if (!this[enable]) {
      return;
    }
    this[theta] += this[delta];
    var local$$54723 = new THREE.Vector3;
    local$$54723[copy](this[camera][position]);
    local$$54723[sub](this[target]);
    /** @type {number} */
    local$$54723[z] = 0;
    var local$$54754 = local$$54723[length]();
    /** @type {number} */
    var local$$54771 = local$$54754 * Math[sin](THREE[Math][degToRad](this[theta]));
    /** @type {number} */
    var local$$54788 = local$$54754 * Math[cos](THREE[Math][degToRad](this[theta]));
    this[camera][position][x] = this[target][x] + local$$54771;
    this[camera][position][y] = this[target][y] + local$$54788;
    this[camera][matrix][lookAt](this[camera][position], this[target], new THREE.Vector3(0, 0, 1));
    this[camera][rotation][setFromRotationMatrix](local$$54464[camera][matrix], local$$54464[camera][rotation][order]);
    this[camera][updateMatrixWorld]();
  };
};
/**
 * @param {?} local$$54896
 * @return {undefined}
 */
LSJFlyToCameraControls = function(local$$54896) {
  this[camera] = local$$54896;
  this[endPosition] = undefined;
  /** @type {number} */
  this[endHeading] = 0;
  /** @type {number} */
  this[endPitch] = 0;
  /** @type {number} */
  this[endRoll] = 0;
  /** @type {number} */
  this[looptime] = 5 * 1E3;
  /** @type {boolean} */
  this[enable] = false;
  /** @type {number} */
  this[startTime] = 0;
  var local$$54946 = this;
  this[vectorKeyframeTrack] = undefined;
  this[quaternionKeyframeTrack] = undefined;
  this[interpolant] = undefined;
  this[quaInterpolant] = undefined;
  /**
   * @param {?} local$$54971
   * @param {?} local$$54972
   * @param {?} local$$54973
   * @param {?} local$$54974
   * @param {?} local$$54975
   * @return {undefined}
   */
  this[flyTo] = function(local$$54971, local$$54972, local$$54973, local$$54974, local$$54975) {
    this[endPosition] = local$$54971;
    this[endHeading] = local$$54972;
    this[endPitch] = local$$54973;
    this[endRoll] = local$$54974;
    this[looptime] = local$$54975;
    /** @type {!Array} */
    var local$$55003 = [];
    /** @type {!Array} */
    var local$$55006 = [];
    local$$55003[push](0);
    this[camera][position][toArray](local$$55006, local$$55006[length]);
    local$$55003[push](this[looptime]);
    local$$54971[toArray](local$$55006, local$$55006[length]);
    this[vectorKeyframeTrack] = new THREE.VectorKeyframeTrack(flytoCamera, local$$55003, local$$55006);
    this[interpolant] = this[vectorKeyframeTrack][createInterpolant](undefined);
    /** @type {!Array} */
    var local$$55066 = [];
    this[camera][quaternion][toArray](local$$55066, local$$55066[length]);
    var local$$55095 = new THREE.Euler(this[endHeading], this[endPitch], this[endRoll], XYZ);
    var local$$55099 = new THREE.Quaternion;
    local$$55099[setFromEuler](local$$55095, true);
    local$$55099[toArray](local$$55066, local$$55066[length]);
    this[quaternionKeyframeTrack] = new THREE.QuaternionKeyframeTrack(flyToCamera, local$$55003, local$$55066);
    this[quaInterpolant] = this[quaternionKeyframeTrack][createInterpolant](undefined);
    /** @type {boolean} */
    this[enable] = true;
    this[startTime] = Date[now]();
  };
  /**
   * @return {undefined}
   */
  this[stop] = function() {
    this[endPosition] = undefined;
    /** @type {number} */
    this[endHeading] = 0;
    /** @type {number} */
    this[endPitch] = 0;
    /** @type {number} */
    this[endRoll] = 0;
    /** @type {number} */
    this[looptime] = 5 * 1E3;
    /** @type {boolean} */
    this[enable] = false;
    /** @type {number} */
    this[startTime] = 0;
  };
  /**
   * @return {undefined}
   */
  this[update] = function() {
    if (!this[enable]) {
      return;
    }
    var local$$55223 = Date[now]();
    if (local$$55223 - this[startTime] > this[looptime]) {
      /** @type {boolean} */
      this[enable] = false;
      local$$54946[camera][position][x] = this[endPosition][x];
      local$$54946[camera][position][y] = this[endPosition][y];
      local$$54946[camera][position][z] = this[endPosition][z];
      return;
    }
    var local$$55305 = this[interpolant][evaluate](local$$55223 - this[startTime]);
    local$$54946[camera][position][fromArray](local$$55305);
    var local$$55329 = this[quaInterpolant][evaluate](local$$55223 - this[startTime]);
    local$$54946[camera][quaternion][fromArray](local$$55329);
    local$$54946[camera][updateMatrixWorld]();
  };
};
/**
 * @param {?} local$$55361
 * @return {undefined}
 */
LSJFlyWithLineControls = function(local$$55361) {
  this[object] = undefined;
  this[tube] = undefined;
  /** @type {boolean} */
  this[enable] = false;
  this[splineCamera] = local$$55361;
  this[binormal] = new THREE.Vector3;
  this[normal] = new THREE.Vector3;
  /** @type {boolean} */
  this[lookAhead] = true;
  /** @type {number} */
  this[looptime] = 20 * 1E3;
  this[startTime] = Date[now]();
  var local$$55421 = this;
  /**
   * @return {?}
   */
  this[toJSON] = function() {
    var local$$55428 = {};
    if (this[object] == undefined) {
      return local$$55428;
    }
    /** @type {!Array} */
    var local$$55439 = [];
    /** @type {number} */
    var local$$55442 = 0;
    for (; local$$55442 < this[object][points][length]; local$$55442++) {
      var local$$55463 = this[object][points][local$$55442];
      local$$55439[push](local$$55463[x], local$$55463[y], local$$55463[z]);
    }
    var local$$55483 = {};
    local$$55483[duration] = this[looptime];
    local$$55483[lineString] = {
      coordinates : local$$55439
    };
    local$$55428[Tour] = {
      name : 线路一,
      playlist : local$$55483
    };
    return local$$55428;
  };
  /**
   * @param {string} local$$55517
   * @param {string} local$$55518
   * @return {undefined}
   */
  this[attach] = function(local$$55517, local$$55518) {
    /** @type {string} */
    this[object] = local$$55517;
    if (local$$55517 != undefined && local$$55421[object][points][length] > 1) {
      this[tube] = new THREE.TubeGeometry(local$$55517, 200, 2, 1, false);
    }
    /** @type {boolean} */
    this[enable] = true;
    if (local$$55518 != undefined) {
      /** @type {string} */
      this[looptime] = local$$55518;
    }
    this[startTime] = Date[now]();
    this[update]();
  };
  /**
   * @return {undefined}
   */
  this[detach] = function() {
    this[object] = undefined;
    this[tube] = undefined;
    /** @type {boolean} */
    this[enable] = false;
  };
  /**
   * @param {number} local$$55613
   * @return {undefined}
   */
  this[update] = function(local$$55613) {
    if (!this[enable]) {
      return;
    }
    if (local$$55421[object] === undefined) {
      return;
    }
    if (this[tube] == undefined && local$$55421[object][points][length] > 1) {
      this[tube] = new THREE.TubeGeometry(this[object], 200, 2, 1, false);
    }
    if (this[tube] === undefined) {
      return;
    }
    var local$$55676 = Date[now]();
    if (local$$55613 < 0 || local$$55613 == 0 || local$$55613 == undefined) {
      /** @type {number} */
      local$$55613 = 1;
    }
    /** @type {number} */
    var local$$55695 = this[looptime] / local$$55613;
    /** @type {number} */
    var local$$55703 = (local$$55676 - this[startTime]) % local$$55695 / local$$55695;
    var local$$55718 = local$$55421[tube][parameters][path][getPointAt](local$$55703);
    var local$$55729 = local$$55421[tube][tangents][length];
    /** @type {number} */
    var local$$55732 = local$$55703 * local$$55729;
    var local$$55738 = Math[floor](local$$55732);
    /** @type {number} */
    var local$$55743 = (local$$55738 + 1) % local$$55729;
    local$$55421[binormal][subVectors](local$$55421[tube][binormals][local$$55743], local$$55421[tube][binormals][local$$55738]);
    local$$55421[binormal][multiplyScalar](local$$55732 - local$$55738)[add](local$$55421[tube][binormals][local$$55738]);
    var local$$55800 = local$$55421[tube][parameters][path][getTangentAt](local$$55703);
    local$$55421[normal][copy](local$$55421[binormal])[cross](local$$55800);
    local$$55421[splineCamera][position][copy](local$$55718);
    /** @type {number} */
    var local$$55844 = local$$55703 + 30 / local$$55421[tube][parameters][path][getLength]();
    /** @type {number} */
    local$$55844 = local$$55844 > 1 ? 1 : local$$55844;
    var local$$55865 = local$$55421[tube][parameters][path][getPointAt](local$$55844);
    local$$55865[z] -= 20;
    local$$55421[splineCamera][matrix][lookAt](local$$55421[splineCamera][position], local$$55865, new THREE.Vector3(0, 0, 1));
    local$$55421[splineCamera][rotation][setFromRotationMatrix](local$$55421[splineCamera][matrix], local$$55421[splineCamera][rotation][order]);
  };
};
/**
 * @return {undefined}
 */
LSJClippingControl = function() {
  /** @type {boolean} */
  this[isRemoved] = false;
  /** @type {boolean} */
  this[bClippingControl] = false;
  /** @type {boolean} */
  this[bDrawing] = false;
  this[meshGroup] = new THREE.Group;
  /** @type {!Array} */
  this[point3ds] = [];
  /** @type {!Array} */
  this[spheres] = [];
  /** @type {number} */
  this[count] = 0;
  this[sphereGeometry] = new THREE.SphereGeometry(.4, 10, 10);
  this[color] = new THREE.Color(16711680);
};
/** @type {function(): undefined} */
LSJClippingControl[prototype][constructor] = LSJClippingControl;
/**
 * @param {?} local$$56012
 * @return {undefined}
 */
LSJClippingControl[prototype][addPoint3D] = function(local$$56012) {
  this[point3ds][push](local$$56012[clone]());
};
/**
 * @param {?} local$$56036
 * @return {undefined}
 */
LSJClippingControl[prototype][setPoint3D] = function(local$$56036) {
  this[point3ds][this[point3ds][length] - 1] = local$$56036;
};
/**
 * @return {undefined}
 */
LSJClippingControl[prototype][clear] = function() {
  /** @type {number} */
  this[count] = 0;
  /** @type {!Array} */
  this[point3ds] = [];
  /** @type {!Array} */
  this[spheres] = [];
  getScene()[remove](this[meshGroup]);
  /** @type {boolean} */
  this[bClippingControl] = false;
  /** @type {boolean} */
  this[bDrawing] = false;
  /** @type {!Array} */
  controlRender[clippingPlanes] = [];
};
/**
 * @param {?} local$$56119
 * @return {undefined}
 */
LSJClippingControl[prototype][update] = function(local$$56119) {
  local$$56119[remove](this[meshGroup]);
  /** @type {!Array} */
  this[spheres] = [];
  this[meshGroup] = new THREE.Group;
  /** @type {number} */
  var local$$56143 = 0;
  for (; local$$56143 < this[point3ds][length]; local$$56143++) {
    var local$$56161 = new THREE.Mesh(this[sphereGeometry], createSphereMaterial());
    var local$$56175 = getCamera()[position][distanceTo](local$$56161[getWorldPosition]());
    var local$$56197 = projectedRadius(1, getCamera()[fov] * Math[PI] / 180, local$$56175, getRenderer()[domElement][clientHeight]);
    /** @type {number} */
    var local$$56201 = 5 / local$$56197;
    local$$56161[position][copy](this[point3ds][local$$56143]);
    local$$56161[scale][set](local$$56201, local$$56201, local$$56201);
    this[spheres][push](local$$56161);
    this[meshGroup][add](local$$56161);
  }
  var local$$56244 = new THREE.Geometry;
  /** @type {number} */
  local$$56143 = 0;
  for (; local$$56143 < this[point3ds][length] - 1; local$$56143++) {
    local$$56244[vertices][push](this[point3ds][local$$56143]);
    local$$56244[vertices][push](this[point3ds][local$$56143 + 1]);
  }
  local$$56244[vertices][push](this[point3ds][0]);
  local$$56244[vertices][push](this[point3ds][this[point3ds][length] - 1]);
  var local$$56327 = new THREE.LineBasicMaterial({
    color : 65280,
    linewidth : 2
  });
  var local$$56331 = new THREE.Line(local$$56244, local$$56327);
  this[meshGroup][add](local$$56331);
  local$$56119[add](this[meshGroup]);
};
/**
 * @param {?} local$$56359
 * @return {undefined}
 */
LSJClippingControl[prototype][updateDragging] = function(local$$56359) {
  local$$56359[remove](this[meshGroup][children][pop]());
  var local$$56378 = new THREE.Geometry;
  /** @type {number} */
  var local$$56381 = 0;
  for (; local$$56381 < this[point3ds][length] - 1; local$$56381++) {
    local$$56378[vertices][push](this[point3ds][local$$56381]);
    local$$56378[vertices][push](this[point3ds][local$$56381 + 1]);
  }
  local$$56378[vertices][push](this[point3ds][0]);
  local$$56378[vertices][push](this[point3ds][this[point3ds][length] - 1]);
  var local$$56461 = new THREE.LineBasicMaterial({
    color : 65280,
    linewidth : 2
  });
  var local$$56465 = new THREE.Line(local$$56378, local$$56461);
  this[meshGroup][children][push](local$$56465);
  local$$56359[add](this[meshGroup]);
};
/**
 * @param {?} local$$56496
 * @return {undefined}
 */
LSJClippingControl[prototype][addMouseMoveListener] = function(local$$56496) {
  if (intersectsObj[length] != 0 && this[bDrawing] == false) {
    if (local$$56496[buttons] == LSJMOUSEBUTTON[right] || local$$56496[buttons] == LSJMOUSEBUTTON[middle]) {
      /** @type {boolean} */
      dragControls[enabled] = false;
    } else {
      /** @type {boolean} */
      dragControls[enabled] = true;
      /** @type {!Array} */
      this[point3ds] = [];
      /** @type {number} */
      var local$$56547 = 0;
      for (; local$$56547 < this[meshGroup][children][length] - 1; local$$56547++) {
        this[point3ds][push](this[meshGroup][children][local$$56547][position]);
      }
      this[updateDragging](getScene());
    }
  }
  if (this[bDrawing] == true) {
    var local$$56608 = intersectSceneAndPlane(local$$56496[clientX], local$$56496[clientY]);
    if (!LSJMath[isZeroVec3](local$$56608)) {
      this[setPoint3D](local$$56608);
      this[update](getScene());
    }
  }
};
/**
 * @param {?} local$$56640
 * @param {?} local$$56641
 * @return {undefined}
 */
LSJClippingControl[prototype][addLeftClickListener] = function(local$$56640, local$$56641) {
  if (this[bClippingControl]) {
    this[count] = this[count] + 1;
    if (this[count] == 1) {
      this[addPoint3D](local$$56641);
      this[addPoint3D](local$$56641);
      /** @type {boolean} */
      this[bDrawing] = true;
    }
    if (this[count] > 1) {
      this[setPoint3D](local$$56641);
      this[addPoint3D](local$$56641);
    }
  }
};
/**
 * @param {?} local$$56709
 * @param {?} local$$56710
 * @return {undefined}
 */
LSJClippingControl[prototype][addDoubleClickListener] = function(local$$56709, local$$56710) {
  dragControls = new THREE.DragControls(controlCamera, this[meshGroup][children], controlRender[domElement]);
  /** @type {boolean} */
  this[bDrawing] = false;
  /** @type {boolean} */
  this[bClippingControl] = false;
};
/**
 * @return {undefined}
 */
LSJClippingControl[prototype][createClippingPlanes] = function() {
  if (this[point3ds][length] > 2) {
    controlRender[clippingPlanes] = this[point3ds];
    controlRender[isRemoved] = this[isRemoved];
  }
  getScene()[remove](this[meshGroup]);
  /** @type {!Array} */
  this[point3ds] = [];
  /** @type {number} */
  this[count] = 0;
  this[meshGroup] = new THREE.Group;
};
/**
 * @return {undefined}
 */
LSJTrackLight = function() {
  /** @type {null} */
  this[light] = null;
  this[position] = new THREE.Vector3;
  /** @type {boolean} */
  this[bTrack] = true;
};
LSJTrackLight[prototype] = Object[create](LSJTrackLight[prototype]);
/** @type {function(): undefined} */
LSJTrackLight[prototype][constructor] = LSJTrackLight;
var intersectsObj;
var controlCamera;
var controlRender;
var controlFeatueLOD;
var controlDiv;
var controlScene;
var controlLayers;
var controlRaycaster;
var controlRaycaster2;
var controlMouse;
var controlMouse2;
var controMouseOffset3d;
var controlNaviIconMesh;
var controlNaviIconPath;
var controlNaviLiveTime;
var controlHovered;
var controlClickStart;
var controlClickEnd;
var bControlStartPan;
var controls;
var stats;
var mouseDWorldPos1;
var mouseDWorldPos2;
var bFirstCamPosSet;
var skyGeometry;
var meshSky;
var lastRenderTime;
var lastControMouseScreen1;
var lastControMouseScreen2;
var movePlaneMesh;
var movePlaneMeshs;
var touchPichPos;
var flyWithLineControls;
var flyToCameraControls;
var flyAroundCenterControls;
var cameraTrackControls;
var editorLineControls;
var controlMomentumParam;
var controlInertiaParam;
var rulerDistance;
var clippingControl;
var dragControls;
var controlSkyBox;
var controlMaxTilt;
var controlMinTilt;
var controlMinZoomDist;
var controlScreenScene;
var controlScreenCam;
var controlLGTextMesh;
var controlNeedStopInertia;
var controlInertiaUsed;
var orbitControls;
var cameraMode;
var sceneCullFace;
var lmModelDefaultMat;
var gdMemUsed;
var gdMaxMemAllowed;
var gdFrameTexMemUsed;
var gdFrameMaxTexMemAllowd;
var billboardPlugin;
/** @type {!Array} */
var billboards = [];
/** @type {!Array} */
var trackLights = [];
var stencilScene;
var pointcloud;
var boundingSphere = new THREE.Sphere;
/** @type {!Array} */
var selectedObjects = [];
var composer;
var effectFXAA;
var outlinePass;
var m_firClickTime;
var m_secClickTime;
var m_timeGap;
var m_lastLBDMouse;
var dControlDoubleClickZoomRatio;
var LSJMOUSEBUTTON = {
  none : 0,
  left : 1,
  right : 2,
  middle : 4
};
/**
 * @param {?} local$$57009
 * @return {?}
 */
function loadTexture(local$$57009) {
  var local$$57013 = new THREE.Texture(local$$57009);
  var local$$57020 = new THREE.MeshBasicMaterial({
    map : local$$57013,
    overdraw : .5
  });
  /** @type {!Image} */
  var local$$57023 = new Image;
  /**
   * @return {undefined}
   */
  local$$57023[onload] = function() {
    local$$57013[image] = this;
    /** @type {boolean} */
    local$$57013[needsUpdate] = true;
  };
  local$$57023[src] = local$$57009;
  return local$$57020;
}
/**
 * @return {?}
 */
function getStats() {
  return stats;
}
/**
 * @return {?}
 */
function getBillboards() {
  return billboards;
}
/**
 * @return {?}
 */
function getLayers() {
  return controlLayers;
}
/**
 * @return {?}
 */
function getFlyWithLineControls() {
  return flyWithLineControls;
}
/**
 * @return {?}
 */
function getFlyToCameraControls() {
  return flyToCameraControls;
}
/**
 * @return {?}
 */
function getEditorLineControls() {
  return editorLineControls;
}
/**
 * @return {?}
 */
function getFlyAroundCenterControls() {
  return flyAroundCenterControls;
}
/**
 * @return {?}
 */
function getCameraTrackControls() {
  return cameraTrackControls;
}
/**
 * @return {?}
 */
function getOrbitControls() {
  return orbitControls;
}
/**
 * @return {?}
 */
function getRulerDistance() {
  return rulerDistance;
}
/**
 * @return {?}
 */
function getScene() {
  return controlScene;
}
/**
 * @return {?}
 */
function getStencilScene() {
  return stencilScene;
}
/**
 * @return {?}
 */
function getCamera() {
  return controlCamera;
}
/**
 * @return {?}
 */
function getRenderer() {
  return controlRender;
}
/**
 * @return {?}
 */
function getSkyBox() {
  return controlSkyBox;
}
/**
 * @param {number} local$$57125
 * @return {undefined}
 */
function setControlMaxTilt(local$$57125) {
  /** @type {number} */
  controlMaxTilt = local$$57125 * Math[PI] / 180;
}
/**
 * @param {number} local$$57138
 * @return {undefined}
 */
function setControlMinZoomDist(local$$57138) {
  /** @type {number} */
  controlMinZoomDist = local$$57138;
}
/**
 * @param {?} local$$57145
 * @return {undefined}
 */
function setControlMinTilt(local$$57145) {
  /** @type {number} */
  controlMinTilt = local$$57145 * Math[PI] / 180;
}
/**
 * @param {number} local$$57158
 * @return {undefined}
 */
function setControlDoubleClickZoomRatio(local$$57158) {
  /** @type {number} */
  dControlDoubleClickZoomRatio = local$$57158;
}
/**
 * @param {!Object} local$$57165
 * @return {undefined}
 */
function setControlInertiaUsed(local$$57165) {
  /** @type {!Object} */
  controlInertiaUsed = local$$57165;
}
/**
 * @return {?}
 */
function getCameraMode() {
  return cameraMode;
}
/**
 * @return {?}
 */
function getMaxMemAllowed() {
  return gdMaxMemAllowed;
}
/**
 * @param {number} local$$57182
 * @return {undefined}
 */
function setMaxMemAllowed(local$$57182) {
  /** @type {number} */
  gdMaxMemAllowed = local$$57182;
}
/**
 * @return {?}
 */
function isSceneCullFace() {
  return sceneCullFace;
}
/**
 * @param {!Object} local$$57194
 * @return {undefined}
 */
function setSceneCullFace(local$$57194) {
  /** @type {!Object} */
  sceneCullFace = local$$57194;
}
/**
 * @param {?} local$$57201
 * @param {?} local$$57202
 * @param {?} local$$57203
 * @param {?} local$$57204
 * @param {?} local$$57205
 * @param {?} local$$57206
 * @param {?} local$$57207
 * @param {?} local$$57208
 * @param {?} local$$57209
 * @param {?} local$$57210
 * @return {undefined}
 */
function addShadowedLight(local$$57201, local$$57202, local$$57203, local$$57204, local$$57205, local$$57206, local$$57207, local$$57208, local$$57209, local$$57210) {
  var local$$57215 = new THREE.DirectionalLight(local$$57204, local$$57205);
  local$$57215[position][set](local$$57201, local$$57202, local$$57203);
  local$$57215[bTrackCamera] = local$$57207;
  controlScene[add](local$$57215);
  local$$57215[castShadow] = local$$57206;
  var local$$57240 = local$$57208;
  /** @type {number} */
  local$$57215[shadowCameraLeft] = -local$$57240;
  local$$57215[shadowCameraRight] = local$$57240;
  local$$57215[shadowCameraTop] = local$$57240;
  /** @type {number} */
  local$$57215[shadowCameraBottom] = -local$$57240;
  local$$57215[shadowCameraNear] = local$$57209;
  local$$57215[shadowCameraFar] = local$$57210;
  /** @type {number} */
  local$$57215[shadowMapWidth] = 1024;
  /** @type {number} */
  local$$57215[shadowMapHeight] = 1024;
  /** @type {number} */
  local$$57215[shadowBias] = -.01;
  if (local$$57206) {
    /** @type {boolean} */
    controlRender[shadowMap][enabled] = true;
    if (sceneCullFace) {
      controlRender[shadowMap][cullFace] = THREE[CullFaceBack];
    }
  }
}
/**
 * @param {number} local$$57319
 * @return {undefined}
 */
function setCameraMode(local$$57319) {
  /** @type {number} */
  cameraMode = local$$57319;
  if (local$$57319 == drag && orbitControls != undefined) {
    orbitControls[dispose]();
    orbitControls == undefined;
  } else {
    if (local$$57319 == orbit) {
      controlCamera[up][set](0, 0, 1);
      orbitControls = new THREE.OrbitControls(controlCamera, controlRender[domElement]);
      /** @type {boolean} */
      orbitControls[enableDamping] = false;
      /** @type {number} */
      orbitControls[dampingFactor] = .35;
      /** @type {boolean} */
      orbitControls[enableZoom] = true;
      setControlMaxTilt(180);
    }
  }
}
/**
 * @return {undefined}
 */
function createModelDefaultMat() {
  lmModelDefaultMat = new THREE.MeshPhongMaterial;
  lmModelDefaultMat[color] = (new THREE.Color)[setRGB](.6, .6, .6);
}
/**
 * @param {number} local$$57407
 * @param {?} local$$57408
 * @return {?}
 */
function initSceneControl(local$$57407, local$$57408) {
  if (!checkLicense(local$$57408)) {
    return false;
  }
  /** @type {number} */
  controlDiv = local$$57407;
  controlScene = new THREE.Scene;
  stencilScene = new THREE.Scene;
  controlScreenScene = new THREE.Scene;
  controlLayers = new LSJLayers;
  controlFeatueLOD = new LSJPageLOD;
  controlCamera = new THREE.PerspectiveCamera(45, window[innerWidth] / window[innerHeight], .1, 15E3);
  controlScreenCam = new THREE.OrthographicCamera(0, window[innerWidth], window[innerHeight], 0, 0, 30);
  /** @type {boolean} */
  sceneCullFace = true;
  createModelDefaultMat();
  /** @type {number} */
  gdMemUsed = 0;
  /** @type {number} */
  gdMaxMemAllowed = 1024 * 1024 * 512;
  /** @type {number} */
  gdFrameTexMemUsed = 0;
  /** @type {number} */
  gdFrameMaxTexMemAllowd = 1024 * 1024 * 8;
  stats = new Stats;
  /** @type {boolean} */
  stats[visble] = false;
  stats[domElement][style][position] = absolute;
  stats[domElement][style][top] = 0px;
  controlDiv[appendChild](stats[domElement]);
  controlSkyBox = new LSJSkyBox;
  controlSkyBox[loadSkyBox](window[innerWidth], window[innerHeight]);
  /** @type {boolean} */
  bControlStartPan = false;
  /** @type {boolean} */
  bFirstCamPosSet = false;
  /** @type {number} */
  lastRenderTime = 0;
  flyToCameraControls = new LSJFlyToCameraControls(controlCamera);
  flyWithLineControls = new LSJFlyWithLineControls(controlCamera);
  flyAroundCenterControls = new LSJFlyAroundCenterControls(controlCamera);
  cameraTrackControls = new LSJCameraTrackControls(controlCamera);
  rulerDistance = new LSJRulerDistance;
  clippingControl = new LSJClippingControl;
  var local$$57582 = new THREE.PlaneBufferGeometry(1, 1, 1, 1);
  var local$$57586 = new THREE.MeshBasicMaterial;
  movePlaneMesh = new THREE.Mesh(local$$57582, local$$57586);
  /** @type {!Array} */
  movePlaneMeshs = [];
  movePlaneMeshs[push](movePlaneMesh);
  controlScene[add](controlFeatueLOD[meshGroup]);
  controlScene[add](controlLayers[meshGroup]);
  controlRender = new THREE.WebGLRenderer({
    antialias : true,
    preserveDrawingBuffer : true
  });
  controlRender[setClearColor](0);
  controlRender[setPixelRatio](window[devicePixelRatio]);
  controlRender[setSize](window[innerWidth], window[innerHeight]);
  /** @type {boolean} */
  controlRender[autoUpdateScene] = true;
  /** @type {boolean} */
  controlRender[autoClearColor] = true;
  billboardPlugin = new LSJBillboardPlugin(controlRender, billboards);
  composer = new THREE.EffectComposer(controlRender);
  renderPass = new THREE.RenderPass(controlScene, controlCamera);
  composer[addPass](renderPass);
  outlinePass = new THREE.OutlinePass(new THREE.Vector2(window[innerWidth], window[innerHeight]), controlScene, controlCamera);
  outlinePass[visibleEdgeColor] = new THREE.Color(.14, .92, .92);
  composer[addPass](outlinePass);
  /**
   * @param {?} local$$57709
   * @return {undefined}
   */
  var local$$57734 = function(local$$57709) {
    outlinePass[patternTexture] = local$$57709;
    local$$57709[wrapS] = THREE[RepeatWrapping];
    local$$57709[wrapT] = THREE[RepeatWrapping];
  };
  var local$$57738 = new THREE.TextureLoader;
  local$$57738[load](textures/tri_pattern.jpg, local$$57734);
  effectFXAA = new THREE.ShaderPass(THREE.FXAAShader);
  effectFXAA[uniforms][resolution][value][set](1 / window[innerWidth], 1 / window[innerHeight]);
  /** @type {boolean} */
  effectFXAA[renderToScreen] = true;
  composer[addPass](effectFXAA);
  cameraMode = drag;
  var local$$57793 = new THREE.Geometry;
  local$$57793[vertices][push](new THREE.Vector3);
  var local$$57815 = new THREE.PointsMaterial({
    size : 40,
    sizeAttenuation : false,
    alphaTest : 0,
    depthTest : false,
    transparent : true
  });
  controlNaviIconPath = resource/image/NaviCursor.png;
  controlNaviIconMesh = new THREE.Points(local$$57793, local$$57815);
  /** @type {boolean} */
  controlNaviIconMesh[visible] = false;
  controlScene[add](controlNaviIconMesh);
  var local$$57839 = new THREE.TextureLoader;
  local$$57839[load](controlNaviIconPath, function(local$$57844) {
    local$$57815[map] = local$$57844;
  });
  controlFeatueLOD[getViewport]()[set](0, 0, window[innerWidth], window[innerHeight]);
  controlDiv[appendChild](controlRender[domElement]);
  editorLineControls = new EditorLineControls(controlCamera, controlScene, controlRender[domElement]);
  window[addEventListener](resize, function local$$57891() {
    /** @type {number} */
    controlCamera[aspect] = window[innerWidth] / window[innerHeight];
    controlScreenCam[right] = window[innerWidth];
    controlScreenCam[top] = window[innerHeight];
    controlCamera[updateProjectionMatrix]();
    controlFeatueLOD[getViewport]()[set](0, 0, window[innerWidth], window[innerHeight]);
    controlRender[setSize](window[innerWidth], window[innerHeight]);
    controlSkyBox[onSceneResize](window[innerWidth], window[innerHeight]);
    var local$$57970 = window[innerWidth] || 1;
    var local$$57977 = window[innerHeight] || 1;
    composer[setSize](local$$57970, local$$57977);
    effectFXAA[uniforms][resolution][value][set](1 / window[innerWidth], 1 / window[innerHeight]);
  }, false);
  /** @type {number} */
  controlMaxTilt = Math[PI] / 2;
  /** @type {number} */
  controlMinTilt = 0;
  /** @type {number} */
  controlMinZoomDist = 20;
  /** @type {number} */
  m_firClickTime = 0;
  /** @type {number} */
  m_secClickTime = 0;
  /** @type {number} */
  m_timeGap = 0;
  /** @type {number} */
  dControlDoubleClickZoomRatio = 1;
  m_lastLBDMouse = new THREE.Vector2;
  /** @type {boolean} */
  controlNeedStopInertia = true;
  /** @type {boolean} */
  controlInertiaUsed = true;
  controlRaycaster = new THREE.Raycaster;
  controlRaycaster2 = new THREE.Raycaster;
  controlMouse = new THREE.Vector2;
  controlMouse2 = new THREE.Vector2;
  lastControMouseScreen1 = new THREE.Vector2;
  lastControMouseScreen2 = new THREE.Vector2;
  controlClickStart = new THREE.Vector2;
  controlClickEnd = new THREE.Vector2;
  controlMomentumParam = new LSJMomentumParam;
  controlInertiaParam = new LSJInertiaParam;
  controMouseOffset3d = new THREE.Vector3;
  mouseDWorldPos1 = new THREE.Vector3;
  mouseDWorldPos2 = new THREE.Vector3;
  touchPichPos = new THREE.Vector3;
  activateSceneControlMouseEvent();
}
/**
 * @return {undefined}
 */
function renderSceneControl() {
  /** @type {!Array} */
  billboards = [];
  /** @type {number} */
  gdFrameTexMemUsed = 0;
  controlFeatueLOD[update](controlCamera);
  if (!bFirstCamPosSet && (!controlFeatueLOD[bdSphere][empty]() || !controlLayers[getBoundingSphere]()[empty]())) {
    var local$$58147 = new THREE.Vector3(0, 0, 0);
    /** @type {number} */
    var local$$58150 = 0;
    if (!controlFeatueLOD[bdSphere][empty]()) {
      this[boundingSphere] = controlFeatueLOD[bdSphere][clone]();
      local$$58147 = controlFeatueLOD[bdSphere][center];
      local$$58150 = controlFeatueLOD[bdSphere][radius];
      controlCamera[position][x] = local$$58147[x];
      controlCamera[position][y] = local$$58147[y];
      controlCamera[position][z] = local$$58147[z] + local$$58150;
    } else {
      this[boundingSphere] = controlLayers[getBoundingSphere]()[clone]();
      local$$58147 = controlLayers[getBoundingSphere]()[center];
      local$$58150 = controlLayers[getBoundingSphere]()[radius];
      controlCamera[position][x] = local$$58147[x];
      controlCamera[position][y] = local$$58147[y];
      controlCamera[position][z] = local$$58147[z] + local$$58150 * 2;
      if (orbitControls != undefined) {
        orbitControls[target] = local$$58147;
      }
    }
    controlCamera[lookAt](new THREE.Vector3(local$$58147[x], local$$58147[y], local$$58147[z]));
    /** @type {boolean} */
    bFirstCamPosSet = true;
    try {
      if (onPageLODLoaded && typeof onPageLODLoaded == function) {
        onPageLODLoaded(local$$58147, local$$58150);
      }
    } catch (local$$58331) {
    }
  }
  local$$58147 = controlLayers[getBoundingSphere]()[center];
  local$$58150 = controlLayers[getBoundingSphere]()[radius];
  if (local$$58150 != 0 && local$$58150 > 100) {
    var local$$58364 = new THREE.Vector3;
    local$$58364[subVectors](controlCamera[position], local$$58147);
    if (local$$58364[length]() - local$$58150 > 0) {
      /** @type {number} */
      controlCamera[near] = 1;
    } else {
      /** @type {number} */
      controlCamera[near] = .1;
    }
    controlCamera[updateProjectionMatrix]();
  }
  rulerDistance[render](this);
  controlLayers[render](this);
  if (outlinePass[selectedObjects][length] > 0) {
    /** @type {boolean} */
    controlRender[autoClear] = true;
    controlRender[setClearColor](16773360);
    controlRender[setClearAlpha](0);
    composer[render]();
  } else {
    controlRender[render](controlScene, controlCamera);
  }
  billboardPlugin[render](controlScene, controlCamera, this);
  stats[update]();
  try {
    if (onProgressInfo && typeof onProgressInfo == function) {
      if (controlFeatueLOD[nodeCount] > 0 && controlFeatueLOD[curLoadingNode] < controlFeatueLOD[nodeCount]) {
        /** @type {number} */
        var local$$58490 = controlFeatueLOD[curLoadingNode] / controlFeatueLOD[nodeCount];
        onProgressInfo(local$$58490);
      }
    }
  } catch (local$$58499) {
  }
}
/**
 * @param {?} local$$58507
 * @return {undefined}
 */
function addFeaturePageLODNode(local$$58507) {
  controlFeatueLOD[addNode](local$$58507);
}
/**
 * @param {?} local$$58517
 * @return {undefined}
 */
function addFeaturePageLODNode1(local$$58517) {
  var local$$58520 = new LSJPageLODNode;
  local$$58520[strDataPath] = local$$58517;
  controlFeatueLOD[addNode](local$$58520);
}
/**
 * @param {?} local$$58535
 * @param {!Object} local$$58536
 * @return {undefined}
 */
function openLFP(local$$58535, local$$58536) {
  if (local$$58536 == undefined) {
    controlFeatueLOD[open](local$$58535);
  } else {
    controlFeatueLOD[fromJson](local$$58535);
  }
}
/**
 * @return {undefined}
 */
function animateSceneControl() {
  requestAnimationFrame(animateSceneControl);
  if (cameraMode == orbit) {
    if (flyAroundCenterControls[enable]) {
      orbitControls[target] = flyAroundCenterControls[target];
    }
    orbitControls[update]();
  }
  flyWithLineControls[update]();
  flyToCameraControls[update]();
  flyAroundCenterControls[update]();
  editorLineControls[update]();
  cameraTrackControls[update]();
  if (cameraMode != orbit) {
    momentumScene();
  }
  inertiaScene();
  var local$$58625 = Date[now]();
  if (controlNaviLiveTime > 0) {
    if (local$$58625 - controlNaviLiveTime > 300) {
      /** @type {boolean} */
      controlNaviIconMesh[visible] = false;
      /** @type {number} */
      controlNaviLiveTime = 0;
    }
  }
  renderSceneControl();
}
/**
 * @param {number} local$$58651
 * @param {number} local$$58652
 * @param {number} local$$58653
 * @param {number} local$$58654
 * @return {undefined}
 */
function doubleTouchStart(local$$58651, local$$58652, local$$58653, local$$58654) {
  stopSceneInertia();
  /** @type {number} */
  m_firClickTime = 0;
  /** @type {number} */
  m_secClickTime = 0;
  mouseDWorldPos1 = intersectSceneAndPlane(local$$58651, local$$58652);
  mouseDWorldPos2 = intersectSceneAndPlane(local$$58653, local$$58654);
  touchPichPos = intersectSceneAndPlane((local$$58651 + local$$58653) / 2, (local$$58652 + local$$58654) / 2);
  /** @type {number} */
  lastControMouseScreen1[x] = local$$58651;
  /** @type {number} */
  lastControMouseScreen1[y] = local$$58652;
  /** @type {number} */
  lastControMouseScreen2[x] = local$$58653;
  /** @type {number} */
  lastControMouseScreen2[y] = local$$58654;
}
/**
 * @param {!Object} local$$58703
 * @return {undefined}
 */
function updateNaviIconMesh(local$$58703) {
  var local$$58708 = controlNaviIconMesh[geometry];
  controlNaviIconMesh[position][copy](local$$58703);
  controlNaviIconMesh[updateMatrixWorld]();
  /** @type {boolean} */
  controlNaviIconMesh[visible] = true;
  /** @type {number} */
  controlNaviLiveTime = 0;
}
/**
 * @param {undefined} local$$58735
 * @param {!Array} local$$58736
 * @param {?} local$$58737
 * @return {undefined}
 */
function mouseDown(local$$58735, local$$58736, local$$58737) {
  flyToCameraControls[stop]();
  flyWithLineControls[detach]();
  flyAroundCenterControls[stop]();
  stopSceneInertia();
  var local$$58757 = intersectSceneAndPlane(local$$58735, local$$58736);
  controlClickStart[set](local$$58735, local$$58736);
  if (!LSJMath[isZeroVec3](local$$58757)) {
    mouseDWorldPos1[set](local$$58757[x], local$$58757[y], local$$58757[z]);
    /** @type {boolean} */
    bControlStartPan = true;
    if (local$$58737 == LSJMOUSEBUTTON[left]) {
      if (updateControlDoubleClick(local$$58735, local$$58736)) {
        controlInertiaParam[vecZoomPos][copy](local$$58757);
        /** @type {boolean} */
        controlInertiaParam[bZoomInertia] = true;
        /** @type {number} */
        controlInertiaParam[nCurZoomTime] = 0;
        controlInertiaParam[dDeltaZoomRatio] = dControlDoubleClickZoomRatio;
        if (onControlPageLODDoubleClick != undefined) {
          var local$$58821 = new THREE.Vector2;
          /** @type {number} */
          local$$58821[x] = local$$58735 / controlRender[domElement][clientWidth] * 2 - 1;
          /** @type {number} */
          local$$58821[y] = -(local$$58736 / controlRender[domElement][clientHeight]) * 2 + 1;
          var local$$58858 = new THREE.Raycaster;
          local$$58858[setFromCamera](local$$58821, controlCamera);
          var local$$58876 = local$$58858[intersectObjects](controlFeatueLOD[meshGroup][children], true);
          /** @type {null} */
          local$$58757 = null;
          /** @type {null} */
          var local$$58882 = null;
          /** @type {number} */
          var local$$58885 = 0;
          /** @type {number} */
          var local$$58888 = 0;
          /** @type {number} */
          var local$$58891 = 0;
          if (local$$58876[length] > 0) {
            local$$58885 = local$$58876[0][point][x];
            local$$58888 = local$$58876[0][point][y];
            local$$58891 = local$$58876[0][point][z];
            local$$58882 = local$$58876[0][object];
            onControlPageLODDoubleClick(new THREE.Vector2(local$$58735, local$$58736), new THREE.Vector3(local$$58885, local$$58888, local$$58891));
          }
        }
      }
    } else {
      if (local$$58737 == LSJMOUSEBUTTON[right] || local$$58737 == LSJMOUSEBUTTON[middle]) {
        updateNaviIconMesh(mouseDWorldPos1);
      }
    }
  } else {
    mouseDWorldPos1[set](0, 0, 0);
  }
  lastControMouseScreen1[x] = local$$58735;
  /** @type {!Array} */
  lastControMouseScreen1[y] = local$$58736;
  if (local$$58737 == LSJMOUSEBUTTON[left]) {
    m_lastLBDMouse[set](local$$58735, local$$58736);
  }
}
/**
 * @param {?} local$$58999
 * @return {undefined}
 */
function onLSJDivMouseDown(local$$58999) {
  local$$58999[preventDefault]();
  if (controlNeedStopInertia) {
    stopSceneInertia();
  }
  mouseDown(local$$58999[clientX], local$$58999[clientY], local$$58999[buttons]);
  try {
    if (onCustomMouseDown && typeof onCustomMouseDown == function) {
      onCustomMouseDown(local$$58999);
    }
  } catch (local$$59033) {
  }
}
/**
 * @param {number} local$$59040
 * @param {!Array} local$$59041
 * @return {?}
 */
function updateControlDoubleClick(local$$59040, local$$59041) {
  if (m_firClickTime == 0) {
    m_firClickTime = Date[now]();
    /** @type {number} */
    m_timeGap = 0;
  } else {
    if (m_secClickTime == 0) {
      m_secClickTime = Date[now]();
      /** @type {number} */
      m_timeGap = m_secClickTime - m_firClickTime;
    }
  }
  var local$$59079 = getDist(local$$59040, local$$59041, m_lastLBDMouse[x], m_lastLBDMouse[y]);
  if (m_timeGap > 0 && m_timeGap < 500) {
    /** @type {number} */
    m_firClickTime = 0;
    /** @type {number} */
    m_secClickTime = 0;
    if (m_lastLBDMouse[x] >= 0 && local$$59040 >= 0 && local$$59079 < 20) {
      return true;
    }
  } else {
    if (m_timeGap >= 500) {
      m_firClickTime = m_secClickTime;
      /** @type {number} */
      m_secClickTime = 0;
    }
  }
  return false;
}
/**
 * @return {undefined}
 */
function mouseUp() {
  /** @type {number} */
  mouseDWorldPos1[x] = 0;
  /** @type {number} */
  mouseDWorldPos1[y] = 0;
  /** @type {number} */
  mouseDWorldPos1[z] = 0;
  /** @type {number} */
  mouseDWorldPos2[x] = 0;
  /** @type {number} */
  mouseDWorldPos2[y] = 0;
  /** @type {number} */
  mouseDWorldPos2[z] = 0;
  /** @type {number} */
  lastControMouseScreen1[x] = -1;
  /** @type {number} */
  lastControMouseScreen1[y] = -1;
  /** @type {number} */
  lastControMouseScreen2[x] = -1;
  /** @type {number} */
  lastControMouseScreen2[y] = -1;
  /** @type {boolean} */
  bControlStartPan = false;
  /** @type {boolean} */
  controlNaviIconMesh[visible] = false;
}
/**
 * @param {?} local$$59198
 * @return {undefined}
 */
function onLSJDivMouseUp(local$$59198) {
  local$$59198[preventDefault]();
  controlClickEnd[set](local$$59198[clientX], local$$59198[clientY]);
  mouseUp();
  if (local$$59198[button] == THREE[MOUSE][LEFT]) {
    doObjectClickEvent(local$$59198[clientX], local$$59198[clientY]);
  }
  if (local$$59198[button] == THREE[MOUSE][RIGHT]) {
    doObjectClickEvent1(local$$59198[clientX], local$$59198[clientY]);
  }
  try {
    if (onCustomMouseUp && typeof onCustomMouseUp == function) {
      onCustomMouseUp(local$$59198);
    }
  } catch (local$$59273) {
  }
}
/**
 * @param {?} local$$59281
 * @param {?} local$$59282
 * @return {?}
 */
function screenToScene(local$$59281, local$$59282) {
  var local$$59286 = new THREE.Vector2;
  /** @type {number} */
  local$$59286[x] = local$$59281 / controlRender[domElement][clientWidth] * 2 - 1;
  /** @type {number} */
  local$$59286[y] = -(local$$59282 / controlRender[domElement][clientHeight]) * 2 + 1;
  var local$$59323 = new THREE.Raycaster;
  local$$59323[setFromCamera](local$$59286, controlCamera);
  var local$$59341 = local$$59323[intersectObjects](controlFeatueLOD[meshGroup][children], true);
  if (local$$59341[length] == 0) {
    local$$59341 = local$$59323[intersectObjects](controlLayers[meshGroup][children], true);
  }
  var local$$59366 = new THREE.Vector3;
  if (local$$59341[length] > 0) {
    if (local$$59341[0] != null && local$$59341[0] != undefined) {
      local$$59366[copy](local$$59341[0][point]);
    }
  }
  return local$$59366;
}
/**
 * @param {string} local$$59400
 * @param {(!Function|RegExp|string)} local$$59401
 * @param {(!Function|RegExp|string)} local$$59402
 * @return {?}
 */
function sceneToScreen(local$$59400, local$$59401, local$$59402) {
  var local$$59406 = new THREE.Vector3(local$$59400, local$$59401, local$$59402);
  local$$59406[project](controlCamera);
  var local$$59421 = new THREE.Vector2(local$$59406[x], local$$59406[y]);
  /** @type {number} */
  local$$59421[x] = (local$$59421[x] * .5 + .5) * controlRender[domElement][clientWidth];
  /** @type {number} */
  local$$59421[y] = controlRender[domElement][clientHeight] - (local$$59421[y] * .5 + .5) * controlRender[domElement][clientHeight];
  return local$$59421;
}
/**
 * @param {number} local$$59472
 * @param {number} local$$59473
 * @return {?}
 */
function intersectScene(local$$59472, local$$59473) {
  /** @type {number} */
  controlMouse[x] = local$$59472 / controlRender[domElement][clientWidth] * 2 - 1;
  /** @type {number} */
  controlMouse[y] = -(local$$59473 / controlRender[domElement][clientHeight]) * 2 + 1;
  controlRaycaster[setFromCamera](controlMouse, controlCamera);
  var local$$59524 = controlRaycaster[intersectObjects](controlFeatueLOD[meshGroup][children], true);
  if (local$$59524[length] == 0) {
    local$$59524 = controlRaycaster[intersectObjects](controlLayers[meshGroup][children], true);
  }
  intersectsObj = controlRaycaster[intersectObjects](clippingControl[meshGroup][children], true);
  if (local$$59524[length] > 0) {
    if (local$$59524[0]) {
      return local$$59524[0][point];
    }
  }
  return new THREE.Vector3;
}
/**
 * @param {number} local$$59583
 * @param {number} local$$59584
 * @return {?}
 */
function intersectSceneAndPlane(local$$59583, local$$59584) {
  var local$$59587 = intersectScene(local$$59583, local$$59584);
  if (LSJMath[isZeroVec3](local$$59587)) {
    local$$59587 = getPageLODCenterPlaneIntersectPos(local$$59583, local$$59584);
  }
  return local$$59587;
}
/**
 * @param {?} local$$59604
 * @return {?}
 */
function getCamAltitude(local$$59604) {
  var local$$59611 = new THREE.Vector3(1, 0, 0);
  var local$$59618 = new THREE.Vector3(0, 1, 0);
  var local$$59626 = (new THREE.Vector3)[copy](local$$59604);
  local$$59626[sub](controlFeatueLOD[bdSphere][center]);
  local$$59626[projectOnPlane](local$$59618);
  local$$59626[projectOnPlane](local$$59611);
  return local$$59626[length]();
}
/**
 * @param {?} local$$59657
 * @return {?}
 */
function getCamXYLen(local$$59657) {
  var local$$59664 = new THREE.Vector3(0, 0, 1);
  var local$$59672 = (new THREE.Vector3)[copy](local$$59657);
  local$$59672[sub](controlFeatueLOD[bdSphere][center]);
  local$$59672[projectOnPlane](local$$59664);
  return local$$59672[length]();
}
/**
 * @param {?} local$$59698
 * @return {?}
 */
function isNewCamAltOutOfRange(local$$59698) {
  var local$$59704 = getCamAltitude(controlCamera[position]);
  var local$$59707 = getCamAltitude(local$$59698);
  if (local$$59707 < local$$59704) {
    return false;
  }
  var local$$59721 = controlFeatueLOD[bdSphere][radius];
  local$$59721 = controlLayers[getBoundingSphere]()[radius] > local$$59721 ? controlLayers[getBoundingSphere]()[radius] : local$$59721;
  /** @type {number} */
  var local$$59743 = local$$59721 * 4;
  if (local$$59707 > local$$59743) {
    return true;
  }
  return false;
}
/**
 * @param {?} local$$59756
 * @return {?}
 */
function isNewCamXYOutOfRange(local$$59756) {
  var local$$59762 = getCamXYLen(controlCamera[position]);
  var local$$59765 = getCamXYLen(local$$59756);
  if (local$$59765 < local$$59762) {
    return false;
  }
  var local$$59779 = controlFeatueLOD[bdSphere][radius];
  local$$59779 = controlLayers[getBoundingSphere]()[radius] > local$$59779 ? controlLayers[getBoundingSphere]()[radius] : local$$59779;
  /** @type {number} */
  var local$$59801 = local$$59779 * 8;
  if (local$$59765 > local$$59801) {
    return true;
  }
  return false;
}
/**
 * @param {?} local$$59814
 * @return {?}
 */
function getCamTilt(local$$59814) {
  var local$$59821 = new THREE.Vector3(-1, 0, 0);
  var local$$59828 = new THREE.Vector3(0, 0, -1);
  var local$$59835 = new THREE.Vector3(0, 0, -1);
  local$$59835[applyQuaternion](local$$59814);
  local$$59821[applyQuaternion](local$$59814);
  var local$$59851 = local$$59835[angleTo](local$$59828);
  var local$$59859 = (new THREE.Vector3)[copy](local$$59835);
  local$$59859[cross](local$$59828)[normalize]();
  var local$$59874 = local$$59859[dot](local$$59821);
  if (local$$59874 < 0) {
    return -local$$59851;
  } else {
    return local$$59851;
  }
}
/**
 * @param {?} local$$59887
 * @return {?}
 */
function isNewCamDirOutOfRange(local$$59887) {
  var local$$59893 = getCamTilt(controlCamera[quaternion]);
  var local$$59896 = getCamTilt(local$$59887);
  if (local$$59893 < controlMinTilt && local$$59896 > local$$59893 && local$$59896 <= Math[PI] / 2) {
    return false;
  }
  if (local$$59893 > controlMaxTilt && local$$59896 < local$$59893 && local$$59896 >= 0) {
    return false;
  }
  if (local$$59896 < controlMinTilt || local$$59896 > controlMaxTilt) {
    return true;
  }
  return false;
}
/**
 * @param {number} local$$59937
 * @param {number} local$$59938
 * @param {?} local$$59939
 * @return {?}
 */
function getPlaneIntersectPos(local$$59937, local$$59938, local$$59939) {
  var local$$59957 = controlFeatueLOD[bdSphere][radius] == 0 ? 100 : controlFeatueLOD[bdSphere][radius];
  local$$59957 = controlLayers[getBoundingSphere]()[radius] > local$$59957 ? controlLayers[getBoundingSphere]()[radius] : local$$59957;
  /** @type {number} */
  movePlaneMesh[scale][x] = local$$59957 * 10;
  /** @type {number} */
  movePlaneMesh[scale][y] = local$$59957 * 10;
  /** @type {number} */
  movePlaneMesh[scale][z] = 1;
  movePlaneMesh[position][copy](local$$59939);
  movePlaneMesh[updateMatrixWorld](true);
  /** @type {number} */
  controlMouse2[x] = local$$59937 / controlRender[domElement][clientWidth] * 2 - 1;
  /** @type {number} */
  controlMouse2[y] = -(local$$59938 / controlRender[domElement][clientHeight]) * 2 + 1;
  controlRaycaster2[setFromCamera](controlMouse2, controlCamera);
  var local$$60063 = controlRaycaster2[intersectObjects](movePlaneMeshs, true);
  if (local$$60063[length] < 1) {
    return new THREE.Vector3;
  }
  return local$$60063[0][point];
}
/**
 * @param {number} local$$60084
 * @param {number} local$$60085
 * @return {?}
 */
function getPageLODCenterPlaneIntersectPos(local$$60084, local$$60085) {
  return getPlaneIntersectPos(local$$60084, local$$60085, controlFeatueLOD[bdSphere][center]);
}
/**
 * @param {?} local$$60098
 * @param {number} local$$60099
 * @return {undefined}
 */
function panSceneDelta(local$$60098, local$$60099) {
  var local$$60107 = (new THREE.Vector3)[copy](local$$60098);
  local$$60107[multiplyScalar](local$$60099);
  var local$$60123 = (new THREE.Vector3)[copy](controlCamera[position]);
  local$$60123[sub](local$$60107);
  if (isNewCamXYOutOfRange(local$$60123)) {
    return;
  }
  controlCamera[position][sub](local$$60107);
  controlCamera[updateMatrixWorld]();
}
/**
 * @param {undefined} local$$60151
 * @param {undefined} local$$60152
 * @return {undefined}
 */
function panSceneInertial(local$$60151, local$$60152) {
  if (!bControlStartPan) {
    return;
  }
  if (LSJMath[isZeroVec2](lastControMouseScreen1)) {
    lastControMouseScreen1[x] = local$$60151;
    lastControMouseScreen1[y] = local$$60152;
    return;
  }
  var local$$60184 = intersectSceneAndPlane(lastControMouseScreen1[x], lastControMouseScreen1[y]);
  if (LSJMath[isZeroVec3](local$$60184)) {
    lastControMouseScreen1[x] = local$$60151;
    lastControMouseScreen1[y] = local$$60152;
    return;
  }
  lastControMouseScreen1[x] = local$$60151;
  lastControMouseScreen1[y] = local$$60152;
  var local$$60215 = getPlaneIntersectPos(local$$60151, local$$60152, local$$60184);
  if (LSJMath[isZeroVec3](local$$60215)) {
    return;
  }
  var local$$60227 = new THREE.Vector3;
  local$$60227[copy](local$$60215)[sub](local$$60184);
  /** @type {number} */
  local$$60227[z] = 0;
  var local$$60253 = (new THREE.Vector3)[copy](controlCamera[position]);
  local$$60253[sub](local$$60227);
  if (isNewCamXYOutOfRange(local$$60253)) {
    return;
  }
  controlCamera[position][sub](local$$60227);
  controlCamera[updateMatrixWorld]();
  controlInertiaParam[vecPanDelta][copy](local$$60227);
  /** @type {boolean} */
  controlInertiaParam[bPanInertia] = true;
  /** @type {number} */
  controlInertiaParam[nCurPanTime] = 0;
  /** @type {number} */
  controlInertiaParam[dDeltaPanRatio] = 1;
}
/**
 * @param {undefined} local$$60307
 * @param {undefined} local$$60308
 * @return {undefined}
 */
function panScene(local$$60307, local$$60308) {
  if (!bControlStartPan) {
    return;
  }
  if (LSJMath[isZeroVec2](lastControMouseScreen1)) {
    lastControMouseScreen1[x] = local$$60307;
    lastControMouseScreen1[y] = local$$60308;
    return;
  }
  var local$$60340 = intersectSceneAndPlane(lastControMouseScreen1[x], lastControMouseScreen1[y]);
  if (LSJMath[isZeroVec3](local$$60340)) {
    lastControMouseScreen1[x] = local$$60307;
    lastControMouseScreen1[y] = local$$60308;
    return;
  }
  lastControMouseScreen1[x] = local$$60307;
  lastControMouseScreen1[y] = local$$60308;
  var local$$60371 = getPlaneIntersectPos(local$$60307, local$$60308, local$$60340);
  if (LSJMath[isZeroVec3](local$$60371)) {
    return;
  }
  var local$$60383 = new THREE.Vector3;
  local$$60383[copy](local$$60371)[sub](local$$60340);
  /** @type {number} */
  local$$60383[z] = 0;
  var local$$60409 = (new THREE.Vector3)[copy](controlCamera[position]);
  local$$60409[sub](local$$60383);
  if (isNewCamXYOutOfRange(local$$60409)) {
    return;
  }
  controlCamera[position][sub](local$$60383);
  controlCamera[updateMatrixWorld]();
}
/**
 * @param {number} local$$60437
 * @param {number} local$$60438
 * @param {number} local$$60439
 * @return {?}
 */
function zoomSceneScreen(local$$60437, local$$60438, local$$60439) {
  var local$$60442 = intersectSceneAndPlane(local$$60437, local$$60438);
  return zoomScene(local$$60442, local$$60439);
}
/**
 * @param {?} local$$60449
 * @param {number} local$$60450
 * @return {?}
 */
function zoomScene(local$$60449, local$$60450) {
  var local$$60458 = (new THREE.Vector3)[copy](local$$60449);
  if (LSJMath[isZeroVec3](local$$60449)) {
    local$$60458[copy](controlFeatueLOD[bdSphere][center]);
  }
  var local$$60480 = new THREE.Vector3;
  local$$60480[copy](local$$60458)[sub](controlCamera[position]);
  var local$$60498 = local$$60480[length]();
  /** @type {number} */
  var local$$60507 = local$$60498 * Math[abs](1 - local$$60450);
  if (local$$60507 < controlMinZoomDist && local$$60450 > 0) {
    return new THREE.Vector3;
  }
  local$$60480[multiplyScalar](local$$60450);
  var local$$60533 = (new THREE.Vector3)[copy](controlCamera[position]);
  local$$60533[add](local$$60480);
  if (isNewCamAltOutOfRange(local$$60533)) {
    return new THREE.Vector3;
  }
  controlCamera[position][add](local$$60480);
  controlCamera[updateMatrixWorld]();
  return local$$60458;
}
/**
 * @param {undefined} local$$60564
 * @param {undefined} local$$60565
 * @param {?} local$$60566
 * @return {undefined}
 */
function rollSceneScreen(local$$60564, local$$60565, local$$60566) {
  var local$$60569 = intersectScene(local$$60564, local$$60565);
  rollSceneAngle(local$$60569, local$$60566 * Math[PI] / 180, true);
}
/**
 * @param {!Object} local$$60583
 * @param {number} local$$60584
 * @param {boolean} local$$60585
 * @return {undefined}
 */
function rollSceneAngle(local$$60583, local$$60584, local$$60585) {
  var local$$60593 = (new THREE.Vector3)[copy](local$$60583);
  if (LSJMath[isZeroVec3](local$$60583)) {
    if (local$$60585) {
      local$$60593[copy](controlFeatueLOD[bdSphere][center]);
    } else {
      return;
    }
  }
  var local$$60626 = (new THREE.Vector3)[copy](controlCamera[position]);
  var local$$60637 = (new THREE.Quaternion)[copy](controlCamera[quaternion]);
  var local$$60652 = (new THREE.Vector3)[copy](local$$60593)[sub](controlCamera[position]);
  var local$$60667 = (new THREE.Quaternion)[copy](controlCamera[quaternion])[inverse]();
  var local$$60675 = (new THREE.Vector3)[copy](local$$60652);
  local$$60675[applyQuaternion](local$$60667);
  var local$$60688 = (new THREE.Vector3)[copy](local$$60675);
  local$$60688[applyQuaternion](local$$60637);
  local$$60626[add](local$$60688);
  var local$$60713 = (new THREE.Vector3(0, 0, 1))[applyQuaternion](local$$60667)[normalize]();
  var local$$60717 = new THREE.Quaternion;
  local$$60717[setFromAxisAngle](local$$60713, local$$60584);
  local$$60637[multiply](local$$60717);
  var local$$60735 = (new THREE.Vector3)[copy](local$$60675);
  local$$60735[multiplyScalar](-1);
  local$$60735[applyQuaternion](local$$60637);
  local$$60626[add](local$$60735);
  controlCamera[quaternion][copy](local$$60637);
  controlCamera[position][copy](local$$60626);
  controlCamera[updateMatrixWorld]();
}
/**
 * @param {!Object} local$$60777
 * @param {number} local$$60778
 * @param {boolean} local$$60779
 * @return {undefined}
 */
function rollScene(local$$60777, local$$60778, local$$60779) {
  /** @type {number} */
  var local$$60792 = local$$60778 * Math[PI] / controlRender[domElement][clientWidth];
  rollSceneAngle(local$$60777, local$$60792, true);
}
/**
 * @param {undefined} local$$60800
 * @param {undefined} local$$60801
 * @param {?} local$$60802
 * @return {undefined}
 */
function pitchScreenScene(local$$60800, local$$60801, local$$60802) {
  var local$$60805 = intersectScene(local$$60800, local$$60801);
  pitchSceneAngle(local$$60805, local$$60802 * Math[PI] / 180);
}
/**
 * @param {!Object} local$$60818
 * @param {number} local$$60819
 * @return {undefined}
 */
function pitchSceneAngle(local$$60818, local$$60819) {
  var local$$60827 = (new THREE.Vector3)[copy](local$$60818);
  if (LSJMath[isZeroVec3](local$$60818)) {
    local$$60827[copy](controlFeatueLOD[bdSphere][center]);
  }
  var local$$60856 = (new THREE.Vector3)[copy](controlCamera[position]);
  var local$$60867 = (new THREE.Quaternion)[copy](controlCamera[quaternion]);
  var local$$60882 = (new THREE.Vector3)[copy](local$$60827)[sub](controlCamera[position]);
  var local$$60897 = (new THREE.Quaternion)[copy](controlCamera[quaternion])[inverse]();
  var local$$60905 = (new THREE.Vector3)[copy](local$$60882);
  local$$60905[applyQuaternion](local$$60897);
  var local$$60918 = (new THREE.Vector3)[copy](local$$60905);
  local$$60918[applyQuaternion](local$$60867);
  local$$60856[add](local$$60918);
  var local$$60935 = new THREE.Vector3(1, 0, 0);
  var local$$60939 = new THREE.Quaternion;
  local$$60939[setFromAxisAngle](local$$60935, local$$60819);
  local$$60867[multiply](local$$60939);
  var local$$60957 = (new THREE.Vector3)[copy](local$$60905);
  local$$60957[multiplyScalar](-1);
  local$$60957[applyQuaternion](local$$60867);
  local$$60856[add](local$$60957);
  if (isNewCamDirOutOfRange(local$$60867)) {
    if (controlInertiaParam[bPitchInteria]) {
      /** @type {boolean} */
      controlInertiaParam[bPitchInteria] = false;
    }
    return;
  }
  controlCamera[quaternion][copy](local$$60867);
  controlCamera[position][copy](local$$60856);
  controlCamera[updateMatrixWorld]();
}
/**
 * @param {!Object} local$$61016
 * @param {number} local$$61017
 * @return {undefined}
 */
function pitchScene(local$$61016, local$$61017) {
  /** @type {number} */
  var local$$61030 = local$$61017 * Math[PI] / controlRender[domElement][clientHeight];
  pitchSceneAngle(local$$61016, local$$61030);
}
/**
 * @param {?} local$$61037
 * @param {?} local$$61038
 * @param {?} local$$61039
 * @return {undefined}
 */
function beginMomentumZoomScreen(local$$61037, local$$61038, local$$61039) {
  var local$$61042 = screenToScene(local$$61037, local$$61038);
  controlMomentumParam[vecZoomPos][copy](local$$61042);
  /** @type {boolean} */
  controlMomentumParam[m_bMomentumZoom] = true;
  controlMomentumParam[dZoomRadio] = local$$61039;
  zoomScene(controlMomentumParam[vecZoomPos], controlMomentumParam[dZoomRadio]);
}
/**
 * @param {?} local$$61074
 * @param {?} local$$61075
 * @param {?} local$$61076
 * @return {undefined}
 */
function beginMomentumRollScreen(local$$61074, local$$61075, local$$61076) {
  var local$$61079 = screenToScene(local$$61074, local$$61075);
  controlMomentumParam[vecRollPos][copy](local$$61079);
  /** @type {boolean} */
  controlMomentumParam[m_bMomentumRoll] = true;
  /** @type {number} */
  controlMomentumParam[dRotateAngle] = local$$61076 * Math[PI] / 180;
  rollSceneAngle(controlMomentumParam[vecRollPos], controlMomentumParam[dRotateAngle]);
}
/**
 * @param {?} local$$61117
 * @param {?} local$$61118
 * @param {?} local$$61119
 * @return {undefined}
 */
function beginMomentumPitchScreen(local$$61117, local$$61118, local$$61119) {
  var local$$61122 = screenToScene(local$$61117, local$$61118);
  controlMomentumParam[vecPitchPos][copy](local$$61122);
  /** @type {boolean} */
  controlMomentumParam[m_bMomentumPitch] = true;
  /** @type {number} */
  controlMomentumParam[dDeltaPitch] = local$$61119 * Math[PI] / 180;
  rollSceneAngle(controlMomentumParam[vecPitchPos], controlMomentumParam[dDeltaPitch]);
}
/**
 * @return {undefined}
 */
function stopSceneMomentum() {
  controlMomentumParam[stop](LSMomentumFlag.MOMENTUM_ALL);
}
/**
 * @return {undefined}
 */
function momentumScene() {
  if (controlMomentumParam[bMomentumRoll]) {
    rollSceneAngle(controlMomentumParam[vecRollPos], controlMomentumParam[dRotateAngle]);
  }
  if (controlMomentumParam[bMomentumPitch]) {
    pitchSceneAngle(controlMomentumParam[vecPitchPos], controlMomentumParam[dDeltaPitch]);
  }
  if (controlMomentumParam[bMomentumZoom]) {
    zoomScene(controlMomentumParam[vecZoomPos], controlMomentumParam[dZoomRadio]);
  }
}
/**
 * @param {?} local$$61216
 * @param {?} local$$61217
 * @return {undefined}
 */
function inertiaPanScene(local$$61216, local$$61217) {
  if (controlInertiaParam[dDeltaPanRatio] / local$$61217 > 0) {
    controlInertiaParam[dDeltaPanRatio] += local$$61217;
  } else {
    controlInertiaParam[dDeltaPanRatio] = local$$61217;
  }
  controlInertiaParam[vecPanDelta][copy](local$$61216);
  /** @type {boolean} */
  controlInertiaParam[bPanInertia] = true;
  /** @type {number} */
  controlInertiaParam[nCurPanTime] = 0;
}
/**
 * @param {?} local$$61262
 * @param {number} local$$61263
 * @return {undefined}
 */
function inertiaZoomScene(local$$61262, local$$61263) {
  if (controlInertiaParam[dDeltaZoomRatio] / local$$61263 > 0) {
    controlInertiaParam[dDeltaZoomRatio] += local$$61263;
  } else {
    /** @type {number} */
    controlInertiaParam[dDeltaZoomRatio] = local$$61263;
  }
  controlInertiaParam[vecZoomPos][copy](local$$61262);
  /** @type {boolean} */
  controlInertiaParam[bZoomInertia] = true;
  /** @type {number} */
  controlInertiaParam[nCurZoomTime] = 0;
}
/**
 * @param {?} local$$61308
 * @param {?} local$$61309
 * @param {?} local$$61310
 * @return {undefined}
 */
function inertiaZoomScreen(local$$61308, local$$61309, local$$61310) {
  var local$$61313 = screenToScene(local$$61308, local$$61309);
  if (cameraMode == orbit) {
    local$$61313 = orbitControls[target];
  }
  controlInertiaParam[vecZoomPos][copy](local$$61313);
  /** @type {boolean} */
  controlInertiaParam[bZoomInertia] = true;
  /** @type {number} */
  controlInertiaParam[nCurZoomTime] = 0;
  controlInertiaParam[dDeltaZoomRatio] = local$$61310;
  /** @type {boolean} */
  controlNeedStopInertia = false;
}
/**
 * @param {!Object} local$$61357
 * @param {number} local$$61358
 * @return {undefined}
 */
function inertiaRollScene(local$$61357, local$$61358) {
  /** @type {number} */
  var local$$61366 = local$$61358 * Math[PI] / 180;
  if (controlInertiaParam[dDeltaRollAngle] / local$$61366 > 0) {
    controlInertiaParam[dDeltaRollAngle] += local$$61366;
  } else {
    /** @type {number} */
    controlInertiaParam[dDeltaRollAngle] = local$$61366;
  }
  controlInertiaParam[vecRollPos][copy](local$$61357);
  /** @type {boolean} */
  controlInertiaParam[bRollInertia] = true;
  /** @type {number} */
  controlInertiaParam[nCurRollTime] = 0;
}
/**
 * @param {?} local$$61411
 * @param {?} local$$61412
 * @param {?} local$$61413
 * @return {undefined}
 */
function inertiaRollScreen(local$$61411, local$$61412, local$$61413) {
  var local$$61416 = screenToScene(local$$61411, local$$61412);
  controlInertiaParam[vecRollPos][copy](local$$61416);
  /** @type {boolean} */
  controlInertiaParam[bRollInertia] = true;
  /** @type {number} */
  controlInertiaParam[nCurRollTime] = 0;
  /** @type {number} */
  controlInertiaParam[dDeltaRollAngle] = local$$61413 * Math[PI] / 180;
  /** @type {boolean} */
  controlNeedStopInertia = true;
}
/**
 * @param {!Object} local$$61455
 * @param {number} local$$61456
 * @return {undefined}
 */
function inertiaPitchScene(local$$61455, local$$61456) {
  /** @type {number} */
  var local$$61464 = local$$61456 * Math[PI] / 180;
  if (controlInertiaParam[dDeltaTilt] / local$$61464 > 0) {
    controlInertiaParam[dDeltaTilt] += local$$61464;
  } else {
    /** @type {number} */
    controlInertiaParam[dDeltaTilt] = local$$61464;
  }
  controlInertiaParam[vecPitchPos][copy](local$$61455);
  /** @type {boolean} */
  controlInertiaParam[bPitchInteria] = true;
  /** @type {number} */
  controlInertiaParam[nCurPitchTime] = 0;
}
/**
 * @param {?} local$$61509
 * @param {?} local$$61510
 * @param {?} local$$61511
 * @return {undefined}
 */
function inertiaPitchScreen(local$$61509, local$$61510, local$$61511) {
  var local$$61514 = screenToScene(local$$61509, local$$61510);
  if (cameraMode == orbit) {
    local$$61514 = orbitControls[target];
  }
  controlInertiaParam[vecPitchPos][copy](local$$61514);
  /** @type {boolean} */
  controlInertiaParam[bPitchInteria] = true;
  /** @type {number} */
  controlInertiaParam[nCurPitchTime] = 0;
  /** @type {number} */
  controlInertiaParam[dDeltaTilt] = local$$61511 * Math[PI] / 180;
  /** @type {boolean} */
  controlNeedStopInertia = true;
}
/**
 * @return {undefined}
 */
function stopSceneInertia() {
  controlInertiaParam[stop](LSInertiaFlag.INERTIA_ALL);
  /** @type {boolean} */
  controlNeedStopInertia = false;
}
/**
 * @return {undefined}
 */
function inertiaScene() {
  if (controlInertiaParam[bPanInertia] && controlInertiaParam[nCurPanTime] < controlInertiaParam[nTotalPanTime]) {
    controlInertiaParam[nCurPanTime]++;
    /** @type {number} */
    var local$$61608 = controlInertiaParam[getInertiaRatio](controlInertiaParam[nCurPanTime], controlInertiaParam[nTotalPanTime]) * controlInertiaParam[dDeltaPanRatio];
    panSceneDelta(controlInertiaParam[vecPanDelta], local$$61608);
  }
  if (controlInertiaParam[bRollInertia] && controlInertiaParam[nCurRollTime] < controlInertiaParam[nTotalRollTime]) {
    controlInertiaParam[nCurRollTime]++;
    /** @type {number} */
    var local$$61648 = controlInertiaParam[getInertiaRatio](controlInertiaParam[nCurRollTime], controlInertiaParam[nTotalRollTime]) * controlInertiaParam[dDeltaRollAngle];
    rollSceneAngle(controlInertiaParam[vecRollPos], local$$61648);
    controlInertiaParam[dDeltaRollAngle] -= local$$61648;
  }
  if (controlInertiaParam[bZoomInertia] && controlInertiaParam[nCurZoomTime] < controlInertiaParam[nTotalZoomTime]) {
    controlInertiaParam[nCurZoomTime]++;
    /** @type {number} */
    var local$$61693 = controlInertiaParam[getInertiaRatio](controlInertiaParam[nCurZoomTime], controlInertiaParam[nTotalZoomTime]) * controlInertiaParam[dDeltaZoomRatio];
    zoomScene(controlInertiaParam[vecZoomPos], local$$61693);
    controlInertiaParam[dDeltaZoomRatio] -= local$$61693;
  }
  if (controlInertiaParam[bPitchInteria] && controlInertiaParam[nCurPitchTime] < controlInertiaParam[nTotalPitchTime]) {
    controlInertiaParam[nCurPitchTime]++;
    /** @type {number} */
    var local$$61738 = controlInertiaParam[getInertiaRatio](controlInertiaParam[nCurPitchTime], controlInertiaParam[nTotalPitchTime]) * controlInertiaParam[dDeltaTilt];
    pitchSceneAngle(controlInertiaParam[vecPitchPos], local$$61738);
    controlInertiaParam[dDeltaTilt] -= local$$61738;
  }
}
/**
 * @param {?} local$$61755
 * @return {undefined}
 */
function onLSJDivMouseMove(local$$61755) {
  local$$61755[preventDefault]();
  if (cameraMode == orbit) {
    return;
  }
  if (local$$61755[buttons] == LSJMOUSEBUTTON[left] && intersectsObj[length] == 0) {
    panSceneInertial(local$$61755[clientX], local$$61755[clientY]);
  } else {
    if (local$$61755[buttons] == LSJMOUSEBUTTON[right] || local$$61755[buttons] == LSJMOUSEBUTTON[middle]) {
      if (LSJMath[isZeroVec2](lastControMouseScreen1)) {
        lastControMouseScreen1[x] = local$$61755[clientX];
        lastControMouseScreen1[y] = local$$61755[clientY];
        return;
      }
      /** @type {number} */
      var local$$61837 = local$$61755[clientX] - lastControMouseScreen1[x];
      /** @type {number} */
      var local$$61846 = local$$61755[clientY] - lastControMouseScreen1[y];
      lastControMouseScreen1[x] = local$$61755[clientX];
      lastControMouseScreen1[y] = local$$61755[clientY];
      if (Math[abs](local$$61837) > Math[abs](local$$61846)) {
        if (controlInertiaUsed) {
          /** @type {number} */
          var local$$61882 = local$$61837 * 360 / controlRender[domElement][clientWidth];
          inertiaRollScene(mouseDWorldPos1, -local$$61882);
        } else {
          rollScene(mouseDWorldPos1, -local$$61837, true);
        }
      } else {
        if (controlInertiaUsed) {
          /** @type {number} */
          local$$61882 = local$$61846 * 180 / controlRender[domElement][clientHeight];
          inertiaPitchScene(mouseDWorldPos1, local$$61882);
        } else {
          pitchScene(mouseDWorldPos1, local$$61846);
        }
      }
    }
  }
  try {
    if (onCustomMouseMove && typeof onCustomMouseMove == function) {
      onCustomMouseMove(local$$61755);
    }
  } catch (local$$61931) {
  }
}
/**
 * @param {?} local$$61939
 * @return {undefined}
 */
function onLSJDivTouchStart(local$$61939) {
  local$$61939[preventDefault]();
  switch(local$$61939[touches][length]) {
    case 1:
      mouseDown(local$$61939[touches][0][clientX], local$$61939[touches][0][clientY], LSJMOUSEBUTTON[left]);
      break;
    case 2:
      doubleTouchStart(local$$61939[touches][0][clientX], local$$61939[touches][0][clientY], local$$61939[touches][1][clientX], local$$61939[touches][1][clientY]);
      break;
    default:
      break;
  }
}
/**
 * @param {?} local$$62022
 * @return {undefined}
 */
function onLSJDivTouchEnd(local$$62022) {
  controlClickEnd[set](local$$62022[changedTouches][0][clientX], local$$62022[changedTouches][0][clientY]);
  mouseUp();
  doObjectClickEvent(controlClickEnd[x], controlClickEnd[y]);
  if (onCustomTouchEnd != undefined) {
    onCustomTouchEnd(local$$62022);
  }
}
/**
 * @param {?} local$$62064
 * @return {undefined}
 */
function onLSJDivTouchCancel(local$$62064) {
  mouseUp();
}
/**
 * @param {number} local$$62070
 * @param {number} local$$62071
 * @param {number} local$$62072
 * @param {number} local$$62073
 * @return {?}
 */
function getDist(local$$62070, local$$62071, local$$62072, local$$62073) {
  /** @type {number} */
  var local$$62076 = local$$62072 - local$$62070;
  /** @type {number} */
  var local$$62079 = local$$62073 - local$$62071;
  return Math[sqrt](local$$62076 * local$$62076 + local$$62079 * local$$62079);
}
/**
 * @param {number} local$$62092
 * @param {number} local$$62093
 * @param {number} local$$62094
 * @param {number} local$$62095
 * @param {number} local$$62096
 * @param {number} local$$62097
 * @param {number} local$$62098
 * @param {number} local$$62099
 * @return {?}
 */
function getCrossPoint(local$$62092, local$$62093, local$$62094, local$$62095, local$$62096, local$$62097, local$$62098, local$$62099) {
  /** @type {null} */
  var local$$62102 = null;
  if (local$$62092 != local$$62094 && local$$62096 != local$$62098) {
    /** @type {number} */
    var local$$62110 = (local$$62093 - local$$62095) / (local$$62092 - local$$62094);
    /** @type {number} */
    var local$$62114 = local$$62093 - local$$62110 * local$$62092;
    /** @type {number} */
    var local$$62119 = (local$$62097 - local$$62099) / (local$$62096 - local$$62098);
    /** @type {number} */
    var local$$62123 = local$$62097 - local$$62119 * local$$62096;
    if (local$$62110 == local$$62119) {
      return false;
    }
    local$$62102 = new THREE.Vector2;
    /** @type {number} */
    local$$62102[x] = -(local$$62114 - local$$62123) / (local$$62110 - local$$62119);
    /** @type {number} */
    local$$62102[y] = (local$$62110 * local$$62123 - local$$62119 * local$$62114) / (local$$62110 - local$$62119);
  } else {
    var local$$62155;
    var local$$62157;
    if (local$$62092 == local$$62094 && local$$62096 != local$$62098) {
      /** @type {number} */
      local$$62155 = (local$$62097 - local$$62099) / (local$$62096 - local$$62098);
      /** @type {number} */
      local$$62157 = local$$62097 - local$$62155 * local$$62096;
      local$$62102 = new THREE.Vector2;
      /** @type {number} */
      local$$62102[x] = local$$62092;
      /** @type {number} */
      local$$62102[y] = local$$62155 * local$$62102[x] + local$$62157;
    } else {
      if (local$$62092 != local$$62094) {
        /** @type {number} */
        local$$62155 = (local$$62093 - local$$62095) / (local$$62092 - local$$62094);
        /** @type {number} */
        local$$62157 = local$$62093 - local$$62155 * local$$62092;
        local$$62102 = new THREE.Vector2;
        /** @type {number} */
        local$$62102[x] = local$$62096;
        /** @type {number} */
        local$$62102[y] = local$$62155 * local$$62102[x] + local$$62157;
      }
    }
  }
  if (local$$62102) {
    if (local$$62102[x] < Math[min](local$$62092, local$$62094) || local$$62102[x] > Math[max](local$$62092, local$$62094) || local$$62102[x] < Math[min](local$$62096, local$$62098) || local$$62102[x] > Math[max](local$$62096, local$$62098) || local$$62102[y] < Math[min](local$$62093, local$$62095) || local$$62102[y] > Math[max](local$$62093, local$$62095) || local$$62102[y] < 
    Math[min](local$$62097, local$$62099) || local$$62102[y] > Math[max](local$$62097, local$$62099)) {
      var local$$62299 = getDist(local$$62092, local$$62093, local$$62094, local$$62095);
      var local$$62302 = getDist(local$$62096, local$$62097, local$$62098, local$$62099);
      if (local$$62299 > local$$62302) {
        return new THREE.Vector2(local$$62096, local$$62097);
      } else {
        return new THREE.Vector2(local$$62092, local$$62093);
      }
    }
  }
  return local$$62102;
}
/**
 * @param {?} local$$62323
 * @return {undefined}
 */
function onLSJDivTouchMove(local$$62323) {
  local$$62323[preventDefault]();
  local$$62323[stopPropagation]();
  if (cameraMode == orbit) {
    return;
  }
  switch(local$$62323[touches][length]) {
    case 1:
      panSceneInertial(local$$62323[touches][0][clientX], local$$62323[touches][0][clientY]);
      lastControMouseScreen1[set](local$$62323[touches][0][clientX], local$$62323[touches][0][clientY]);
      break;
    case 2:
      var local$$62400 = local$$62323[touches][0][clientX];
      var local$$62410 = local$$62323[touches][0][clientY];
      var local$$62420 = local$$62323[touches][1][clientX];
      var local$$62430 = local$$62323[touches][1][clientY];
      if (lastControMouseScreen1[x] >= 0 && lastControMouseScreen2[x] >= 0) {
        var local$$62450 = getDist(lastControMouseScreen1[x], lastControMouseScreen1[y], local$$62400, local$$62410);
        var local$$62459 = getDist(lastControMouseScreen2[x], lastControMouseScreen2[y], local$$62420, local$$62430);
        var local$$62474 = getDist(lastControMouseScreen1[x], lastControMouseScreen1[y], lastControMouseScreen2[x], lastControMouseScreen2[y]);
        var local$$62477 = getDist(local$$62400, local$$62410, local$$62420, local$$62430);
        var local$$62491 = Math[acos]((lastControMouseScreen1[x] - lastControMouseScreen2[x]) / local$$62474);
        var local$$62499 = Math[acos]((local$$62400 - local$$62420) / local$$62477);
        if (local$$62459 >= 0 || local$$62450 >= 0) {
          var local$$62514 = Math[abs](local$$62400 - lastControMouseScreen1[x]);
          var local$$62524 = Math[abs](local$$62410 - lastControMouseScreen1[y]);
          var local$$62534 = Math[abs](local$$62420 - lastControMouseScreen2[x]);
          var local$$62544 = Math[abs](local$$62430 - lastControMouseScreen2[y]);
          /** @type {number} */
          var local$$62547 = 0;
          if (local$$62400 < local$$62420 && local$$62410 < local$$62430 || local$$62400 > local$$62420 && local$$62410 < local$$62430) {
            /** @type {number} */
            local$$62547 = local$$62491 - local$$62499;
          } else {
            /** @type {number} */
            local$$62547 = -(local$$62491 - local$$62499);
          }
          if (Math[abs](local$$62547) < .01) {
            /** @type {number} */
            var local$$62577 = local$$62410 - lastControMouseScreen1[y];
            /** @type {number} */
            var local$$62583 = local$$62430 - lastControMouseScreen2[y];
            /** @type {number} */
            var local$$62589 = local$$62400 - lastControMouseScreen1[x];
            /** @type {number} */
            var local$$62595 = local$$62420 - lastControMouseScreen2[x];
            if (local$$62589 * local$$62595 > 0) {
              /** @type {number} */
              local$$62595 = local$$62595 == 0 ? 1 : local$$62595;
              /** @type {number} */
              var local$$62607 = local$$62589 / local$$62595;
              rollSceneAngle(touchPichPos, local$$62607 * Math[min](local$$62589, local$$62595) * .01, true);
            }
          } else {
            rollSceneAngle(touchPichPos, local$$62547, true);
          }
          if (local$$62524 > local$$62514 && local$$62544 > local$$62534 && local$$62524 > 3 && local$$62544 > 3 && local$$62577 * local$$62583 > 0 && Math[abs](local$$62474 - local$$62477) < 5) {
            /** @type {number} */
            var local$$62650 = local$$62577 / local$$62524;
            pitchScene(touchPichPos, local$$62650 * Math[min](local$$62524, local$$62544));
          }
          var local$$62678 = Math[min](controlRender[domElement][clientHeight], controlRender[domElement][clientWidth]);
          /** @type {number} */
          var local$$62684 = 2 * (local$$62477 - local$$62474) / local$$62678;
          zoomSceneScreen((local$$62400 + local$$62420) / 2, (local$$62410 + local$$62430) / 2, local$$62684);
        }
      }
      lastControMouseScreen1[set](local$$62400, local$$62410);
      lastControMouseScreen2[set](local$$62420, local$$62430);
      break;
    default:
      break;
  }
}
/**
 * @param {?} local$$62719
 * @return {undefined}
 */
function addSelectedObject(local$$62719) {
  if (local$$62719[type] == GeoModelLOD) {
    local$$62719[layer][addSelectionObject](local$$62719);
    outlinePass[selectedObjects][push](local$$62719[meshGroup]);
  }
}
/**
 * @return {undefined}
 */
function releaseSelectedObject() {
  getLayers()[releaseSelection]();
  /** @type {!Array} */
  outlinePass[selectedObjects] = [];
}
/**
 * @param {?} local$$62766
 * @param {?} local$$62767
 * @return {undefined}
 */
function doObjectClickEvent(local$$62766, local$$62767) {
  var local$$62780 = Math[abs](controlClickEnd[x] - controlClickStart[x]);
  var local$$62793 = Math[abs](controlClickEnd[y] - controlClickStart[y]);
  if (local$$62780 < 2 && local$$62793 < 2) {
    if (onControlPageLODClick != undefined) {
      var local$$62804 = new THREE.Vector2;
      /** @type {number} */
      local$$62804[x] = local$$62766 / controlRender[domElement][clientWidth] * 2 - 1;
      /** @type {number} */
      local$$62804[y] = -(local$$62767 / controlRender[domElement][clientHeight]) * 2 + 1;
      var local$$62841 = new THREE.Raycaster;
      local$$62841[setFromCamera](local$$62804, controlCamera);
      var local$$62859 = local$$62841[intersectObjects](controlFeatueLOD[meshGroup][children], true);
      /** @type {null} */
      var local$$62862 = null;
      /** @type {null} */
      var local$$62865 = null;
      /** @type {number} */
      var local$$62868 = 0;
      /** @type {number} */
      var local$$62871 = 0;
      /** @type {number} */
      var local$$62874 = 0;
      if (local$$62859[length] > 0) {
        local$$62868 = local$$62859[0][point][x];
        local$$62871 = local$$62859[0][point][y];
        local$$62874 = local$$62859[0][point][z];
        local$$62865 = local$$62859[0][object];
        onControlPageLODClick(new THREE.Vector2(local$$62766, local$$62767), new THREE.Vector3(local$$62868, local$$62871, local$$62874));
      }
    }
    if (onControlFeatureClick != undefined) {
      local$$62804 = new THREE.Vector2;
      /** @type {number} */
      local$$62804[x] = local$$62766 / controlRender[domElement][clientWidth] * 2 - 1;
      /** @type {number} */
      local$$62804[y] = -(local$$62767 / controlRender[domElement][clientHeight]) * 2 + 1;
      local$$62841 = new THREE.Raycaster;
      local$$62841[setFromCamera](local$$62804, controlCamera);
      local$$62859 = local$$62841[intersectObjects](controlLayers[meshGroup][children], true);
      /** @type {null} */
      local$$62862 = null;
      /** @type {null} */
      local$$62865 = null;
      /** @type {number} */
      local$$62868 = 0;
      /** @type {number} */
      local$$62871 = 0;
      /** @type {number} */
      local$$62874 = 0;
      releaseSelectedObject();
      if (local$$62859[length] > 0) {
        local$$62868 = local$$62859[0][point][x];
        local$$62871 = local$$62859[0][point][y];
        local$$62874 = local$$62859[0][point][z];
        local$$62865 = local$$62859[0][object];
        if (local$$62865 != null && local$$62865 != undefined) {
          var local$$63056 = local$$62865[parent];
          for (; local$$63056 != undefined && local$$63056[type] == Group && local$$63056[Owner] == undefined;) {
            local$$63056 = local$$63056[parent];
          }
          if (local$$63056 != undefined && local$$63056[Owner] != undefined && local$$63056[Owner][type] == GeoModelLOD) {
            local$$63056[Owner][layer][addSelectionObject](local$$63056.Owner);
            outlinePass[selectedObjects][push](local$$63056);
            onControlFeatureClick(new THREE.Vector2(local$$62766, local$$62767), new THREE.Vector3(local$$62868, local$$62871, local$$62874), local$$63056.Owner);
          } else {
            if (local$$62865[type] != GeoLabel && local$$62865[type] != GeoMarker) {
              outlinePass[selectedObjects][push](local$$62859[0][object]);
            }
            onControlFeatureClick(new THREE.Vector2(local$$62766, local$$62767), new THREE.Vector3(local$$62868, local$$62871, local$$62874), local$$62865);
          }
        }
      }
    }
  }
}
/**
 * @param {?} local$$63172
 * @param {?} local$$63173
 * @return {undefined}
 */
function doObjectClickEvent1(local$$63172, local$$63173) {
  var local$$63186 = Math[abs](controlClickEnd[x] - controlClickStart[x]);
  var local$$63199 = Math[abs](controlClickEnd[y] - controlClickStart[y]);
  if (local$$63186 < 2 && local$$63199 < 2) {
    if (onControlPageLODRightClick != undefined) {
      var local$$63210 = new THREE.Vector2;
      /** @type {number} */
      local$$63210[x] = local$$63172 / controlRender[domElement][clientWidth] * 2 - 1;
      /** @type {number} */
      local$$63210[y] = -(local$$63173 / controlRender[domElement][clientHeight]) * 2 + 1;
      var local$$63247 = new THREE.Raycaster;
      local$$63247[setFromCamera](local$$63210, controlCamera);
      var local$$63265 = local$$63247[intersectObjects](controlFeatueLOD[meshGroup][children], true);
      /** @type {null} */
      var local$$63268 = null;
      /** @type {null} */
      var local$$63271 = null;
      /** @type {number} */
      var local$$63274 = 0;
      /** @type {number} */
      var local$$63277 = 0;
      /** @type {number} */
      var local$$63280 = 0;
      if (local$$63265[length] > 0) {
        local$$63274 = local$$63265[0][point][x];
        local$$63277 = local$$63265[0][point][y];
        local$$63280 = local$$63265[0][point][z];
        local$$63271 = local$$63265[0][object];
        onControlPageLODRightClick(new THREE.Vector2(local$$63172, local$$63173), new THREE.Vector3(local$$63274, local$$63277, local$$63280));
      }
    }
  }
}
/**
 * @param {?} local$$63339
 * @return {undefined}
 */
function onLSJDivMouseWheel(local$$63339) {
  flyToCameraControls[stop]();
  flyWithLineControls[detach]();
  flyAroundCenterControls[stop]();
  local$$63339[preventDefault]();
  if (cameraMode == orbit) {
    return;
  }
  /** @type {number} */
  var local$$63369 = 0;
  if (local$$63339[wheelDelta]) {
    /** @type {number} */
    local$$63369 = local$$63339[wheelDelta] / 40;
  } else {
    if (local$$63339[detail]) {
      /** @type {number} */
      local$$63369 = -local$$63339[detail] / 3;
    }
  }
  /** @type {number} */
  var local$$63400 = local$$63369 / 30;
  if (controlNeedStopInertia) {
    stopSceneInertia();
  }
  var local$$63414 = intersectSceneAndPlane(local$$63339[clientX], local$$63339[clientY]);
  if (!LSJMath[isZeroVec3](local$$63414)) {
    if (controlInertiaUsed) {
      inertiaZoomScene(local$$63414, local$$63400);
    } else {
      zoomScene(local$$63414, local$$63400);
    }
    updateNaviIconMesh(local$$63414);
    controlNaviLiveTime = Date[now]();
  }
}
/**
 * @param {?} local$$63442
 * @return {undefined}
 */
function onLSJContextmenu(local$$63442) {
  local$$63442[preventDefault]();
}
/**
 * @return {undefined}
 */
function activateSceneControlMouseEvent() {
  controlDiv[addEventListener](mousemove, onLSJDivMouseMove, false);
  controlDiv[addEventListener](mousedown, onLSJDivMouseDown, false);
  controlDiv[addEventListener](mouseup, onLSJDivMouseUp, false);
  controlDiv[addEventListener](mousewheel, onLSJDivMouseWheel, false);
  controlDiv[addEventListener](contextmenu, onLSJContextmenu, false);
  controlDiv[addEventListener](MozMousePixelScroll, onLSJDivMouseWheel, false);
  controlDiv[addEventListener](touchstart, onLSJDivTouchStart, false);
  controlDiv[addEventListener](touchend, onLSJDivTouchEnd, false);
  controlDiv[addEventListener](touchcancel, onLSJDivTouchCancel, false);
  controlDiv[addEventListener](touchmove, onLSJDivTouchMove, false);
}
/**
 * @return {undefined}
 */
function dectivateSceneControlMouseEvent() {
  controlDiv[removeEventListener](mousemove, onLSJDivMouseMove, false);
  controlDiv[removeEventListener](mousedown, onLSJDivMouseDown, false);
  controlDiv[removeEventListener](mouseup, onLSJDivMouseUp, false);
  controlDiv[removeEventListener](mousewheel, onLSJDivMouseWheel, false);
  controlDiv[removeEventListener](contextmenu, onLSJContextmenu, false);
  controlDiv[removeEventListener](MozMousePixelScroll, onLSJDivMouseWheel, false);
  controlDiv[removeEventListener](touchstart, onLSJDivTouchStart, false);
  controlDiv[removeEventListener](touchend, onLSJDivTouchEnd, false);
  controlDiv[removeEventListener](touchcancel, onLSJDivTouchCancel, false);
  controlDiv[removeEventListener](touchmove, onLSJDivTouchMove, false);
}
var DisplayMode = {
  None : 0,
  Wireframe : 1,
  Heightmap : 2,
  HeightmapWireframe : 3
};
var displayMode = DisplayMode[None];
/**
 * @param {?} local$$63628
 * @param {!Object} local$$63629
 * @return {undefined}
 */
var setDisplayMode = function(local$$63628, local$$63629) {
  switch(local$$63628) {
    case DisplayMode[None]:
      displayMode = DisplayMode[None];
      /** @type {null} */
      controlScene[overrideMaterial] = null;
      break;
    case DisplayMode[Wireframe]:
      displayMode = DisplayMode[Wireframe];
      /** @type {null} */
      controlScene[overrideMaterial] = null;
      break;
    case DisplayMode[Heightmap]:
      {
        displayMode = DisplayMode[Heightmap];
        var local$$63690 = varying float height; + void main()  + { + gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); + height = position.z; + };
        var local$$63727 = _0x34b6[2E3] + 
 + _0x34b6[2001] + _0x34b6[2002] + _0x34b6[2003] + _0x34b6[2004] + _0x34b6[2005] + _0x34b6[2006] + _0x34b6[2007] + _0x34b6[2008] + _0x34b6[2009] + _0x34b6[2010];
        var local$$63732 = {
          colorRange : {
            value : local$$63629
          }
        };
        controlScene[overrideMaterial] = new THREE.ShaderMaterial({
          uniforms : local$$63732,
          vertexShader : local$$63690,
          fragmentShader : local$$63727
        });
        break;
      }
    case DisplayMode[_0x34b6[2011]]:
      {
        displayMode = DisplayMode[_0x34b6[2011]];
        local$$63690 = varying float height; + void main()  + { + gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); + height = position.z; + };
        local$$63727 = _0x34b6[2E3] + 
 + _0x34b6[2001] + _0x34b6[2002] + _0x34b6[2003] + _0x34b6[2004] + _0x34b6[2005] + _0x34b6[2006] + _0x34b6[2007] + _0x34b6[2008] + _0x34b6[2009] + _0x34b6[2010];
        local$$63732 = {
          colorRange : {
            value : local$$63629
          }
        };
        controlScene[overrideMaterial] = new THREE.ShaderMaterial({
          uniforms : local$$63732,
          vertexShader : local$$63690,
          fragmentShader : local$$63727,
          wireframe : true
        });
        break;
      }
  }
};
var LSMomentumFlag = {
  MOMENTUM_NONE : 0,
  MOMENTUM_PAN : 1,
  MOMENTUM_ROLL : 2,
  MOMENTUM_PITCH : 4,
  MOMENTUM_ZOOM : 8,
  MOMENTUM_ALL : 15
};
var LSInertiaFlag = {
  INERTIA_NONE : 0,
  INERTIA_PAN : 1,
  INERTIA_ROLL : 2,
  INERTIA_PITCH : 4,
  INERTIA_ZOOM : 8,
  INERTIA_ALL : 15
};
/**
 * @return {undefined}
 */
LSJMomentumParam = function() {
  /** @type {boolean} */
  this[bMomentumRoll] = false;
  /** @type {boolean} */
  this[bMomentumPitch] = false;
  /** @type {boolean} */
  this[bMomentumZoom] = false;
  this[vecRollPos] = new THREE.Vector3;
  this[vecPitchPos] = new THREE.Vector3;
  this[vecZoomPos] = new THREE.Vector3;
  /** @type {number} */
  this[dRotateAngle] = 0;
  /** @type {number} */
  this[dDeltaPitch] = 0;
  /** @type {number} */
  this[dZoomRadio] = 0;
};
/** @type {function(): undefined} */
LSJMomentumParam[prototype][constructor] = LSJMomentumParam;
/**
 * @param {?} local$$63930
 * @return {?}
 */
LSJMomentumParam[prototype][_0x34b6[2012]] = function(local$$63930) {
  if (local$$63930 & LSMomentumFlag[_0x34b6[2013]]) {
    return this[dRotateAngle];
  }
  if (local$$63930 & LSMomentumFlag[_0x34b6[2014]]) {
    return this[dDeltaPitch];
  }
  if (local$$63930 & LSMomentumFlag[_0x34b6[2015]]) {
    return this[dZoomRadio];
  }
  return 0;
};
/**
 * @param {?} local$$63977
 * @return {undefined}
 */
LSJMomentumParam[prototype][start] = function(local$$63977) {
  if (local$$63977 & LSMomentumFlag[_0x34b6[2013]]) {
    /** @type {boolean} */
    this[bMomentumRoll] = true;
  }
  if (local$$63977 & LSMomentumFlag[_0x34b6[2014]]) {
    /** @type {boolean} */
    this[bMomentumPitch] = true;
  }
  if (local$$63977 & LSMomentumFlag[_0x34b6[2015]]) {
    /** @type {boolean} */
    this[bMomentumZoom] = true;
  }
};
/**
 * @param {?} local$$64027
 * @return {undefined}
 */
LSJMomentumParam[prototype][stop] = function(local$$64027) {
  if (local$$64027 & LSMomentumFlag[_0x34b6[2013]]) {
    /** @type {boolean} */
    this[bMomentumRoll] = false;
    /** @type {number} */
    this[dRotateAngle] = 0;
  }
  if (local$$64027 & LSMomentumFlag[_0x34b6[2014]]) {
    /** @type {boolean} */
    this[bMomentumPitch] = false;
    /** @type {number} */
    this[dDeltaPitch] = 0;
  }
  if (local$$64027 & LSMomentumFlag[_0x34b6[2015]]) {
    /** @type {boolean} */
    this[bMomentumZoom] = false;
    /** @type {number} */
    this[dZoomRadio] = 0;
  }
};
/**
 * @return {undefined}
 */
LSJInertiaParam = function() {
  /** @type {boolean} */
  this[bPitchInteria] = false;
  /** @type {boolean} */
  this[bZoomInertia] = false;
  /** @type {boolean} */
  this[bRollInertia] = false;
  /** @type {boolean} */
  this[bPanInertia] = false;
  /** @type {number} */
  this[nCurPitchTime] = 0;
  /** @type {number} */
  this[nCurZoomTime] = 0;
  /** @type {number} */
  this[nCurRollTime] = 0;
  /** @type {number} */
  this[nCurPanTime] = 0;
  /** @type {number} */
  this[nTotalPanTime] = 20;
  /** @type {number} */
  this[nTotalZoomTime] = 30;
  /** @type {number} */
  this[nTotalRollTime] = 20;
  /** @type {number} */
  this[nTotalPitchTime] = 10;
  /** @type {number} */
  this[dDeltaTilt] = 0;
  /** @type {number} */
  this[dDeltaRollAngle] = 0;
  /** @type {number} */
  this[dDeltaZoomRatio] = 0;
  /** @type {number} */
  this[dDeltaPanRatio] = 0;
  this[vecRollPos] = new THREE.Vector3;
  this[vecPitchPos] = new THREE.Vector3;
  this[vecZoomPos] = new THREE.Vector3;
  this[vecPanDelta] = new THREE.Vector3;
};
/** @type {function(): undefined} */
LSJInertiaParam[prototype][constructor] = LSJInertiaParam;
/**
 * @param {?} local$$64232
 * @param {number} local$$64233
 * @return {?}
 */
LSJInertiaParam[prototype][getInertiaRatio] = function(local$$64232, local$$64233) {
  return (2 * (local$$64233 + 1 - local$$64232) - 1) / (local$$64233 * local$$64233);
};
/**
 * @param {?} local$$64255
 * @return {undefined}
 */
LSJInertiaParam[prototype][start] = function(local$$64255) {
  if (local$$64255 & LSInertiaFlag[_0x34b6[2016]]) {
    /** @type {boolean} */
    this[bPanInertia] = true;
  }
  if (local$$64255 & LSInertiaFlag[_0x34b6[2017]]) {
    /** @type {boolean} */
    this[_0x34b6[2018]] = true;
  }
  if (local$$64255 & LSInertiaFlag[_0x34b6[2019]]) {
    /** @type {boolean} */
    this[bPitchInteria] = true;
  }
  if (local$$64255 & LSInertiaFlag[_0x34b6[2020]]) {
    /** @type {boolean} */
    this[_0x34b6[2021]] = true;
  }
};
/**
 * @param {?} local$$64318
 * @return {undefined}
 */
LSJInertiaParam[prototype][stop] = function(local$$64318) {
  if (local$$64318 & LSInertiaFlag[_0x34b6[2016]]) {
    /** @type {boolean} */
    this[bPanInertia] = false;
    /** @type {number} */
    this[nCurPanTime] = 0;
    /** @type {number} */
    this[dDeltaPanRatio] = 0;
  }
  if (local$$64318 & LSInertiaFlag[_0x34b6[2017]]) {
    /** @type {boolean} */
    this[_0x34b6[2018]] = false;
    /** @type {number} */
    this[nCurRollTime] = 0;
    /** @type {number} */
    this[dDeltaRollAngle] = 0;
  }
  if (local$$64318 & LSInertiaFlag[_0x34b6[2019]]) {
    /** @type {boolean} */
    this[bPitchInteria] = false;
    /** @type {number} */
    this[nCurPitchTime] = 0;
    /** @type {number} */
    this[dDeltaTilt] = 0;
  }
  if (local$$64318 & LSMomentumFlag[_0x34b6[2020]]) {
    /** @type {boolean} */
    this[_0x34b6[2021]] = false;
    /** @type {number} */
    this[nCurZoomTime] = 0;
    /** @type {number} */
    this[dDeltaZoomRatio] = 0;
  }
};
/**
 * @return {undefined}
 */
LSJRulerDistance = function() {
  this[_0x34b6[2022]] = undefined;
  this[_0x34b6[2023]] = undefined;
  this[_0x34b6[2024]] = undefined;
  this[_0x34b6[2025]] = undefined;
  this[_0x34b6[2026]] = undefined;
  this[_0x34b6[2027]] = undefined;
  this[_0x34b6[2028]] = undefined;
  this[LineSegments] = undefined;
  this[meshGroup] = new THREE.Group;
  /** @type {boolean} */
  this[_0x34b6[2029]] = true;
  /** @type {!Array} */
  this[point3ds] = [];
  /** @type {!Array} */
  this[spheres] = [];
  /** @type {!Array} */
  this[_0x34b6[2030]] = [];
  this[sphereGeometry] = new THREE.SphereGeometry(.4, 5, 5);
  this[color] = new THREE.Color(16711680);
};
/** @type {function(): undefined} */
LSJRulerDistance[prototype][constructor] = LSJRulerDistance;
/**
 * @param {?} local$$64533
 * @return {undefined}
 */
LSJRulerDistance[prototype][addPoint3D] = function(local$$64533) {
  if (this[point3ds][length] < 2) {
    this[point3ds][push](local$$64533[clone]());
    /** @type {boolean} */
    this[_0x34b6[2029]] = true;
  }
};
/**
 * @param {?} local$$64573
 * @param {?} local$$64574
 * @return {undefined}
 */
LSJRulerDistance[prototype][setPoint3D] = function(local$$64573, local$$64574) {
  if (this[point3ds][length] > local$$64573) {
    this[point3ds][local$$64573][copy](local$$64574);
    /** @type {boolean} */
    this[_0x34b6[2029]] = true;
  }
};
/**
 * @return {undefined}
 */
LSJRulerDistance[prototype][clear] = function() {
  /** @type {!Array} */
  this[point3ds] = [];
  getScene()[remove](this[meshGroup]);
  this[_0x34b6[2022]] = undefined;
  this[_0x34b6[2023]] = undefined;
  this[_0x34b6[2024]] = undefined;
  this[_0x34b6[2025]] = undefined;
  this[_0x34b6[2026]] = undefined;
  this[_0x34b6[2027]] = undefined;
  this[LineSegments] = undefined;
  this[_0x34b6[2028]] = undefined;
};
/**
 * @return {?}
 */
var createSphereMaterial = function() {
  var local$$64678 = new THREE.MeshLambertMaterial({
    color : 16711680,
    ambient : 11184810,
    depthTest : false,
    depthWrite : false
  });
  return local$$64678;
};
/**
 * @param {?} local$$64690
 * @return {undefined}
 */
LSJRulerDistance[prototype][update] = function(local$$64690) {
  if (this[point3ds][length] == 2) {
    local$$64690[remove](this[meshGroup]);
    /** @type {!Array} */
    this[_0x34b6[2030]] = [];
    /** @type {!Array} */
    this[spheres] = [];
    this[meshGroup] = new THREE.Group;
    var local$$64732 = this[point3ds][0];
    var local$$64739 = this[point3ds][1];
    var local$$64743 = new THREE.Vector3;
    if (local$$64732[z] < local$$64739[z]) {
      local$$64743[x] = local$$64732[x];
      local$$64743[y] = local$$64732[y];
      local$$64743[z] = local$$64739[z];
    } else {
      local$$64743[x] = local$$64739[x];
      local$$64743[y] = local$$64739[y];
      local$$64743[z] = local$$64732[z];
    }
    var local$$64806 = new THREE.Geometry;
    local$$64806[vertices][push](local$$64732[clone]());
    local$$64806[vertices][push](local$$64743);
    local$$64806[vertices][push](local$$64743);
    local$$64806[vertices][push](local$$64739[clone]());
    local$$64806[_0x34b6[2031]]();
    this[_0x34b6[2028]] = new THREE.LineDashedMaterial({
      color : 65280,
      dashSize : .5,
      gapSize : .5,
      linewidth : 2
    });
    this[LineSegments] = new THREE.LineSegments(local$$64806, this[_0x34b6[2028]]);
    this[meshGroup][add](this.LineSegments);
    var local$$64889 = new THREE.Geometry;
    local$$64889[vertices][push](local$$64732[clone]());
    local$$64889[vertices][push](local$$64739[clone]());
    var local$$64920 = new THREE.LineBasicMaterial({
      color : 65280,
      linewidth : 2
    });
    var local$$64924 = new THREE.Line(local$$64889, local$$64920);
    this[meshGroup][add](local$$64924);
    var local$$64936 = new THREE.Geometry;
    local$$64936[vertices][push](local$$64732[clone]());
    local$$64936[vertices][push](local$$64743);
    local$$64936[vertices][push](local$$64739[clone]());
    var local$$64976 = new THREE.PointsMaterial({
      color : 16711680,
      size : 1,
      depthTest : false
    });
    var local$$64980 = new THREE.Points(local$$64936, local$$64976);
    var local$$64988 = new THREE.Mesh(this[sphereGeometry], createSphereMaterial());
    local$$64988[position][copy](local$$64732);
    var local$$65010 = getCamera()[position][distanceTo](local$$64988[getWorldPosition]());
    var local$$65031 = projectedRadius(1, getCamera()[fov] * Math[PI] / 180, local$$65010, getRenderer()[domElement][clientHeight]);
    /** @type {number} */
    var local$$65035 = 10 / local$$65031;
    local$$64988[scale][set](local$$65035, local$$65035, local$$65035);
    this[spheres][push](local$$64988);
    this[meshGroup][add](local$$64988);
    var local$$65067 = new THREE.Mesh(this[sphereGeometry], createSphereMaterial());
    local$$65067[position][copy](local$$64743);
    local$$65067[scale][set](local$$65035, local$$65035, local$$65035);
    this[spheres][push](local$$65067);
    this[meshGroup][add](local$$65067);
    var local$$65107 = new THREE.Mesh(this[sphereGeometry], createSphereMaterial());
    local$$65107[position][copy](local$$64739);
    local$$65107[scale][set](local$$65035, local$$65035, local$$65035);
    this[spheres][push](local$$65107);
    this[meshGroup][add](local$$65107);
    this[_0x34b6[2025]] = new LSJGeoMarker;
    var local$$65148 = new LSJMarkerStyle;
    /** @type {number} */
    local$$65148[iconSize] = 25;
    var local$$65160 = local$$65148[getTextStyle]();
    local$$65160[setFontName](_0x34b6[2032]);
    local$$65160[getFillColor]()[setRGB](1, 1, 1);
    local$$65160[setStrokeWidth](20);
    this[_0x34b6[2025]][setStyle](local$$65148);
    local$$65010 = local$$64739[clone]()[sub](local$$64732)[length]();
    this[_0x34b6[2025]][setName](local$$65010[_0x34b6[2033]](2) + _0x34b6[2034]);
    this[_0x34b6[2025]][setPosition](local$$64732[x] + (local$$64739[x] - local$$64732[x]) / 2, local$$64732[y] + (local$$64739[y] - local$$64732[y]) / 2, local$$64732[z] + (local$$64739[z] - local$$64732[z]) / 2);
    var local$$65274 = new LSJTextSprite;
    local$$65274[_0x34b6[2035]]({
      r : 0,
      g : 0,
      b : 0,
      a : .8
    });
    local$$65274[_0x34b6[2036]]({
      r : 0,
      g : 0,
      b : 0,
      a : .3
    });
    local$$65274[_0x34b6[2037]](local$$65010[_0x34b6[2033]](2) + _0x34b6[2034]);
    /** @type {boolean} */
    local$$65274[material][depthTest] = false;
    local$$65274[updateMatrixWorld]();
    /** @type {boolean} */
    local$$65274[visible] = true;
    var local$$65370 = new THREE.Vector3(local$$64732[x] + (local$$64739[x] - local$$64732[x]) / 2, local$$64732[y] + (local$$64739[y] - local$$64732[y]) / 2, local$$64732[z] + (local$$64739[z] - local$$64732[z]) / 2);
    local$$65274[position][copy](local$$65370);
    local$$65010 = getCamera()[position][distanceTo](local$$65274[getWorldPosition]());
    local$$65031 = projectedRadius(1, getCamera()[fov] * Math[PI] / 180, local$$65010, getRenderer()[domElement][clientHeight]);
    /** @type {number} */
    local$$65035 = 60 / local$$65031;
    local$$65274[scale][set](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[2030]][push](local$$65274);
    this[meshGroup][add](local$$65274);
    this[_0x34b6[2026]] = new LSJGeoMarker;
    var local$$65450 = new LSJMarkerStyle;
    /** @type {number} */
    local$$65450[iconSize] = 25;
    var local$$65462 = local$$65450[getTextStyle]();
    local$$65462[setFontName](_0x34b6[2032]);
    local$$65462[getFillColor]()[setRGB](1, 1, 1);
    local$$65462[setStrokeWidth](20);
    this[_0x34b6[2026]][setStyle](local$$65450);
    var local$$65497 = undefined;
    if (local$$64732[z] < local$$64739[z]) {
      local$$65497 = local$$64732[clone]()[sub](local$$64743);
    } else {
      local$$65497 = local$$64739[clone]()[sub](local$$64743);
    }
    this[_0x34b6[2026]][setName](local$$65497[length]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    this[_0x34b6[2026]][setPosition](local$$64743[x] + local$$65497[x] / 2, local$$64743[y] + local$$65497[y] / 2, local$$64743[z] + local$$65497[z] / 2);
    var local$$65586 = new LSJTextSprite;
    local$$65586[_0x34b6[2035]]({
      r : 0,
      g : 0,
      b : 0,
      a : .8
    });
    local$$65586[_0x34b6[2036]]({
      r : 0,
      g : 0,
      b : 0,
      a : .3
    });
    local$$65586[_0x34b6[2037]](local$$65497[length]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    /** @type {boolean} */
    local$$65586[material][depthTest] = false;
    /** @type {boolean} */
    local$$65586[visible] = true;
    var local$$65669 = new THREE.Vector3(local$$64743[x] + local$$65497[x] / 2, local$$64743[y] + local$$65497[y] / 2, local$$64743[z] + local$$65497[z] / 2);
    local$$65586[position][copy](local$$65669);
    local$$65010 = getCamera()[position][distanceTo](local$$65586[getWorldPosition]());
    local$$65031 = projectedRadius(1, getCamera()[fov] * Math[PI] / 180, local$$65010, getRenderer()[domElement][clientHeight]);
    /** @type {number} */
    local$$65035 = 60 / local$$65031;
    local$$65586[scale][set](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[2030]][push](local$$65586);
    this[meshGroup][add](local$$65586);
    this[_0x34b6[2027]] = new LSJGeoMarker;
    var local$$65749 = new LSJMarkerStyle;
    /** @type {number} */
    local$$65749[iconSize] = 30;
    var local$$65761 = local$$65749[getTextStyle]();
    local$$65761[setFontName](_0x34b6[2032]);
    local$$65761[getFillColor]()[setRGB](1, 1, 1);
    local$$65761[setStrokeWidth](20);
    this[_0x34b6[2027]][setStyle](local$$65749);
    if (local$$64732[z] < local$$64739[z]) {
      local$$65497 = local$$64739[clone]()[sub](local$$64743);
    } else {
      local$$65497 = local$$64732[clone]()[sub](local$$64743);
    }
    this[_0x34b6[2027]][setName](local$$65497[length]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    this[_0x34b6[2027]][setPosition](local$$64743[x] + local$$65497[x] / 2, local$$64743[y] + local$$65497[y] / 2, local$$64743[z] + local$$65497[z] / 2);
    var local$$65883 = new LSJTextSprite;
    local$$65883[_0x34b6[2035]]({
      r : 0,
      g : 0,
      b : 0,
      a : .8
    });
    local$$65883[_0x34b6[2036]]({
      r : 0,
      g : 0,
      b : 0,
      a : .3
    });
    local$$65883[_0x34b6[2037]](local$$65497[length]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    /** @type {boolean} */
    local$$65883[material][depthTest] = false;
    /** @type {boolean} */
    local$$65883[visible] = true;
    var local$$65966 = new THREE.Vector3(local$$64743[x] + local$$65497[x] / 2, local$$64743[y] + local$$65497[y] / 2, local$$64743[z] + local$$65497[z] / 2);
    local$$65883[position][copy](local$$65966);
    local$$65010 = getCamera()[position][distanceTo](local$$65883[getWorldPosition]());
    local$$65031 = projectedRadius(1, getCamera()[fov] * Math[PI] / 180, local$$65010, getRenderer()[domElement][clientHeight]);
    /** @type {number} */
    local$$65035 = 60 / local$$65031;
    local$$65883[scale][set](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[2030]][push](local$$65883);
    this[meshGroup][add](local$$65883);
    local$$64690[add](this[meshGroup]);
  }
  /** @type {boolean} */
  this[_0x34b6[2029]] = false;
};
/**
 * @param {?} local$$66066
 * @return {undefined}
 */
LSJRulerDistance[prototype][render] = function(local$$66066) {
  if (this[_0x34b6[2029]]) {
    this[update](local$$66066[_0x34b6[2038]]());
  }
  if (this[_0x34b6[2028]] != undefined) {
    if (this[LineSegments][geometry][boundingSphere] === null) {
      this[LineSegments][geometry][computeBoundingSphere]();
    }
    var local$$66124 = this[LineSegments][geometry][boundingSphere][center];
    var local$$66137 = local$$66124[clone]()[project](local$$66066[controlCamera]);
    /** @type {number} */
    var local$$66156 = (local$$66137[x] + 1) / 2 * local$$66066[controlRender][domElement][clientWidth];
    /** @type {number} */
    var local$$66176 = -(local$$66137[y] - 1) / 2 * local$$66066[controlRender][domElement][clientHeight];
    /** @type {number} */
    local$$66137[x] = (local$$66156 + 1) / controlRender[domElement][clientWidth] * 2 - 1;
    local$$66137[unproject](local$$66066[controlCamera]);
    var local$$66212 = local$$66137[sub](local$$66124)[length]();
    /** @type {number} */
    var local$$66227 = this[LineSegments][geometry][boundingSphere][radius] / local$$66212;
    /** @type {number} */
    var local$$66231 = local$$66227 / 6;
    /** @type {number} */
    var local$$66246 = this[LineSegments][geometry][boundingSphere][radius] / local$$66231;
    /** @type {number} */
    var local$$66248 = local$$66246;
    this[_0x34b6[2028]][setValues]({
      dashSize : local$$66246,
      gapSize : local$$66248,
      linewidth : 3
    });
  }
  /** @type {number} */
  var local$$66264 = 0;
  for (; local$$66264 < this[_0x34b6[2030]][length]; local$$66264++) {
    var local$$66279 = this[_0x34b6[2030]][local$$66264];
    var local$$66295 = local$$66066[controlCamera][position][distanceTo](local$$66279[getWorldPosition]());
    var local$$66317 = projectedRadius(1, local$$66066[controlCamera][fov] * Math[PI] / 180, local$$66295, controlRender[domElement][clientHeight]);
    /** @type {number} */
    var local$$66321 = 60 / local$$66317;
    local$$66279[scale][set](local$$66321, local$$66321, local$$66321);
  }
  /** @type {number} */
  var local$$66335 = 0;
  for (; local$$66335 < this[spheres][length]; local$$66335++) {
    var local$$66350 = this[spheres][local$$66335];
    local$$66295 = local$$66066[controlCamera][position][distanceTo](local$$66350[getWorldPosition]());
    local$$66317 = projectedRadius(1, local$$66066[controlCamera][fov] * Math[PI] / 180, local$$66295, controlRender[domElement][clientHeight]);
    /** @type {number} */
    local$$66321 = 10 / local$$66317;
    local$$66350[scale][set](local$$66321, local$$66321, local$$66321);
  }
};
(function() {
  /**
   * @param {?} local$$66412
   * @return {undefined}
   */
  function local$$66411(local$$66412) {
    this[object] = local$$66412;
    this[target] = new THREE.Vector3;
    /** @type {number} */
    this[_0x34b6[2039]] = 0;
    /** @type {number} */
    this[_0x34b6[2040]] = Infinity;
    /** @type {number} */
    this[_0x34b6[2041]] = 0;
    /** @type {number} */
    this[_0x34b6[2042]] = Infinity;
    /** @type {number} */
    this[_0x34b6[2043]] = -Infinity;
    /** @type {number} */
    this[_0x34b6[2044]] = Infinity;
    /** @type {number} */
    this[_0x34b6[2045]] = -Infinity;
    /** @type {number} */
    this[_0x34b6[2046]] = Infinity;
    /** @type {boolean} */
    this[enableDamping] = false;
    /** @type {number} */
    this[dampingFactor] = .005;
    var local$$66483 = this;
    /** @type {number} */
    var local$$66486 = 1E-6;
    /** @type {number} */
    var local$$66489 = 0;
    /** @type {number} */
    var local$$66492 = .1;
    /** @type {number} */
    var local$$66495 = 0;
    /** @type {number} */
    var local$$66498 = 0;
    /** @type {number} */
    var local$$66501 = 1;
    var local$$66505 = new THREE.Vector3;
    var local$$66509 = new THREE.Vector3;
    /** @type {boolean} */
    var local$$66512 = false;
    /**
     * @return {undefined}
     */
    this[_0x34b6[2047]] = function() {
      var local$$66535 = (new THREE.Quaternion)[_0x34b6[2048]](this[object][up], new THREE.Vector3(0, 1, 0));
      var local$$66545 = local$$66535[clone]()[inverse]();
      var local$$66553 = this[object][position];
      var local$$66557 = new THREE.Vector3;
      local$$66557[copy](local$$66553)[sub](this[target]);
      local$$66557[applyQuaternion](local$$66535);
      local$$66489 = Math[atan2](local$$66557[x], local$$66557[z]);
      local$$66492 = Math[atan2](Math[sqrt](local$$66557[x] * local$$66557[x] + local$$66557[z] * local$$66557[z]), local$$66557[y]);
    };
    /**
     * @return {?}
     */
    this[_0x34b6[2049]] = function() {
      return local$$66492;
    };
    /**
     * @return {?}
     */
    this[_0x34b6[2050]] = function() {
      return local$$66489;
    };
    /**
     * @param {?} local$$66641
     * @return {undefined}
     */
    this[_0x34b6[2051]] = function(local$$66641) {
      /** @type {number} */
      local$$66498 = local$$66498 - local$$66641;
    };
    /**
     * @param {?} local$$66653
     * @return {undefined}
     */
    this[_0x34b6[2052]] = function(local$$66653) {
      /** @type {number} */
      local$$66495 = local$$66495 - local$$66653;
    };
    this[_0x34b6[2053]] = function() {
      var local$$66668 = new THREE.Vector3;
      return function local$$66670(local$$66671) {
        var local$$66682 = this[object][matrix][elements];
        local$$66668[set](local$$66682[0], local$$66682[1], local$$66682[2]);
        local$$66668[multiplyScalar](-local$$66671);
        local$$66505[add](local$$66668);
      };
    }();
    this[_0x34b6[2054]] = function() {
      var local$$66720 = new THREE.Vector3;
      return function local$$66722(local$$66723) {
        var local$$66734 = this[object][matrix][elements];
        local$$66720[set](local$$66734[4], local$$66734[5], local$$66734[6]);
        local$$66720[multiplyScalar](local$$66723);
        local$$66505[add](local$$66720);
      };
    }();
    /**
     * @param {number} local$$66768
     * @param {number} local$$66769
     * @param {number} local$$66770
     * @param {number} local$$66771
     * @return {undefined}
     */
    this[_0x34b6[2055]] = function(local$$66768, local$$66769, local$$66770, local$$66771) {
      if (local$$66483[object] instanceof THREE[_0x34b6[2056]]) {
        var local$$66786 = local$$66483[object][position];
        var local$$66799 = local$$66786[clone]()[sub](local$$66483[target]);
        var local$$66805 = local$$66799[length]();
        /** @type {number} */
        local$$66805 = local$$66805 * Math[tan](local$$66483[object][fov] / 2 * Math[PI] / 180);
        local$$66483[_0x34b6[2053]](2 * local$$66768 * local$$66805 / local$$66771);
        local$$66483[_0x34b6[2054]](2 * local$$66769 * local$$66805 / local$$66771);
      } else {
        if (local$$66483[object] instanceof THREE[_0x34b6[2057]]) {
          local$$66483[_0x34b6[2053]](local$$66768 * (local$$66483[object][right] - local$$66483[object][left]) / local$$66770);
          local$$66483[_0x34b6[2054]](local$$66769 * (local$$66483[object][top] - local$$66483[object][bottom]) / local$$66771);
        } else {
          console[warn](_0x34b6[2058]);
        }
      }
    };
    /**
     * @param {?} local$$66913
     * @return {undefined}
     */
    this[_0x34b6[2059]] = function(local$$66913) {
      if (local$$66483[object] instanceof THREE[_0x34b6[2056]]) {
        /** @type {number} */
        local$$66501 = local$$66501 / local$$66913;
      } else {
        if (local$$66483[object] instanceof THREE[_0x34b6[2057]]) {
          local$$66483[object][_0x34b6[2060]] = Math[max](this[_0x34b6[2041]], Math[min](this[_0x34b6[2042]], this[object][_0x34b6[2060]] * local$$66913));
          local$$66483[object][updateProjectionMatrix]();
          /** @type {boolean} */
          local$$66512 = true;
        } else {
          console[warn](_0x34b6[2061]);
        }
      }
    };
    /**
     * @param {?} local$$66992
     * @return {undefined}
     */
    this[_0x34b6[2062]] = function(local$$66992) {
      if (local$$66483[object] instanceof THREE[_0x34b6[2056]]) {
        /** @type {number} */
        local$$66501 = local$$66501 * local$$66992;
      } else {
        if (local$$66483[object] instanceof THREE[_0x34b6[2057]]) {
          local$$66483[object][_0x34b6[2060]] = Math[max](this[_0x34b6[2041]], Math[min](this[_0x34b6[2042]], this[object][_0x34b6[2060]] / local$$66992));
          local$$66483[object][updateProjectionMatrix]();
          /** @type {boolean} */
          local$$66512 = true;
        } else {
          console[warn](_0x34b6[2061]);
        }
      }
    };
    this[update] = function() {
      var local$$67074 = new THREE.Vector3;
      var local$$67090 = (new THREE.Quaternion)[_0x34b6[2048]](local$$66412[up], new THREE.Vector3(0, 1, 0));
      var local$$67100 = local$$67090[clone]()[inverse]();
      var local$$67104 = new THREE.Vector3;
      var local$$67108 = new THREE.Quaternion;
      return function() {
        var local$$67117 = this[object][position];
        local$$67074[copy](local$$67117)[sub](this[target]);
        local$$67074[applyQuaternion](local$$67090);
        local$$66489 = local$$66489 + local$$66498;
        local$$66492 = local$$66492 + local$$66495;
        local$$66489 = Math[max](this[_0x34b6[2045]], Math[min](this[_0x34b6[2046]], local$$66489));
        local$$66492 = Math[max](this[_0x34b6[2043]], Math[min](this[_0x34b6[2044]], local$$66492));
        /** @type {number} */
        var local$$67179 = local$$67074[length]() * local$$66501;
        local$$67179 = Math[max](this[_0x34b6[2039]], Math[min](this[_0x34b6[2040]], local$$67179));
        this[target][add](local$$66505);
        /** @type {number} */
        local$$67074[x] = local$$67179 * Math[sin](local$$66492) * Math[sin](local$$66489);
        /** @type {number} */
        local$$67074[y] = local$$67179 * Math[cos](local$$66492);
        /** @type {number} */
        local$$67074[z] = local$$67179 * Math[sin](local$$66492) * Math[cos](local$$66489);
        local$$67074[applyQuaternion](local$$67100);
        local$$67117[copy](this[target])[add](local$$67074);
        var local$$67267 = new THREE.Vector3(0, 1, 0);
        var local$$67274 = new THREE.Vector3(1, 0, 0);
        local$$67267[_0x34b6[2063]](local$$67274, local$$66492);
        if (local$$67267[z] < 0) {
          /** @type {number} */
          this[object][up][z] = -1;
          this[object][lookAt](this[target]);
        } else {
          /** @type {number} */
          this[object][up][z] = 1;
          this[object][lookAt](this[target]);
        }
        if (this[enableDamping] === true) {
          /** @type {number} */
          local$$66498 = local$$66498 * (1 - this[dampingFactor]);
          /** @type {number} */
          local$$66495 = local$$66495 * (1 - this[dampingFactor]);
        } else {
          /** @type {number} */
          local$$66498 = 0;
          /** @type {number} */
          local$$66495 = 0;
        }
        /** @type {number} */
        local$$66501 = 1;
        local$$66505[set](0, 0, 0);
        if (local$$66512 || local$$67104[_0x34b6[2064]](this[object][position]) > local$$66486 || 8 * (1 - local$$67108[dot](this[object][quaternion])) > local$$66486) {
          local$$67104[copy](this[object][position]);
          local$$67108[copy](this[object][quaternion]);
          /** @type {boolean} */
          local$$66512 = false;
          return true;
        }
        return false;
      };
    }();
  }
  /**
   * @param {?} local$$67451
   * @param {!Object} local$$67452
   * @return {undefined}
   */
  THREE[_0x34b6[2065]] = function(local$$67451, local$$67452) {
    /**
     * @param {number} local$$67455
     * @param {number} local$$67456
     * @return {undefined}
     */
    function local$$67454(local$$67455, local$$67456) {
      var local$$67473 = local$$67458[domElement] === document ? local$$67458[domElement][body] : local$$67458[domElement];
      local$$67475[_0x34b6[2055]](local$$67455, local$$67456, local$$67473[clientWidth], local$$67473[clientHeight]);
    }
    /**
     * @return {?}
     */
    function local$$67489() {
      return 2 * Math[PI] / 60 / 60 * local$$67458[_0x34b6[2074]];
    }
    /**
     * @return {?}
     */
    function local$$67507() {
      return Math[_0x34b6[2084]](.95, local$$67458[_0x34b6[2068]]);
    }
    /**
     * @param {?} local$$67521
     * @return {undefined}
     */
    function local$$67520(local$$67521) {
      if (local$$67458[enabled] === false) {
        return;
      }
      local$$67521[preventDefault]();
      if (local$$67521[button] === local$$67458[_0x34b6[2077]][_0x34b6[2085]]) {
        if (local$$67458[_0x34b6[2069]] === false) {
          return;
        }
        local$$67556 = local$$67557[_0x34b6[2086]];
        local$$67563[set](local$$67521[clientX], local$$67521[clientY]);
      } else {
        if (local$$67521[button] === local$$67458[_0x34b6[2077]][_0x34b6[2087]]) {
          if (local$$67458[_0x34b6[2071]] === false) {
            return;
          }
          local$$67556 = local$$67557[_0x34b6[2088]];
          local$$67600[set](local$$67521[clientX], local$$67521[clientY]);
        } else {
          if (local$$67521[button] === local$$67458[_0x34b6[2077]][_0x34b6[2088]]) {
            if (local$$67458[_0x34b6[2071]] === false) {
              return;
            }
            local$$67556 = local$$67557[_0x34b6[2088]];
            local$$67600[set](local$$67521[clientX], local$$67521[clientY]);
          }
        }
      }
      if (local$$67556 !== local$$67557[_0x34b6[2079]]) {
        document[addEventListener](mousemove, local$$67664, false);
        document[addEventListener](mouseup, local$$67673, false);
        local$$67458[dispatchEvent](local$$67680);
      }
    }
    /**
     * @param {?} local$$67687
     * @return {undefined}
     */
    function local$$67664(local$$67687) {
      if (local$$67458[enabled] === false) {
        return;
      }
      local$$67687[preventDefault]();
      var local$$67717 = local$$67458[domElement] === document ? local$$67458[domElement][body] : local$$67458[domElement];
      if (local$$67556 === local$$67557[_0x34b6[2086]]) {
        if (local$$67458[_0x34b6[2069]] === false) {
          return;
        }
        local$$67732[set](local$$67687[clientX], local$$67687[clientY]);
        local$$67744[subVectors](local$$67732, local$$67563);
        local$$67475[_0x34b6[2051]](2 * Math[PI] * local$$67744[x] / local$$67717[clientWidth] * local$$67458[_0x34b6[2070]]);
        local$$67475[_0x34b6[2052]](2 * Math[PI] * local$$67744[y] / local$$67717[clientHeight] * local$$67458[_0x34b6[2070]]);
        local$$67563[copy](local$$67732);
      } else {
        if (local$$67556 === local$$67557[_0x34b6[2089]]) {
          if (local$$67458[enableZoom] === false) {
            return;
          }
          local$$67813[set](local$$67687[clientX], local$$67687[clientY]);
          local$$67825[subVectors](local$$67813, local$$67829);
          if (local$$67825[y] > 0) {
            local$$67475[_0x34b6[2059]](local$$67507());
          } else {
            if (local$$67825[y] < 0) {
              local$$67475[_0x34b6[2062]](local$$67507());
            }
          }
          local$$67829[copy](local$$67813);
        } else {
          if (local$$67556 === local$$67557[_0x34b6[2088]]) {
            if (local$$67458[_0x34b6[2071]] === false) {
              return;
            }
            local$$67879[set](local$$67687[clientX], local$$67687[clientY]);
            local$$67891[subVectors](local$$67879, local$$67600);
            local$$67454(local$$67891[x], local$$67891[y]);
            local$$67600[copy](local$$67879);
          }
        }
      }
      if (local$$67556 !== local$$67557[_0x34b6[2079]]) {
        local$$67458[update]();
      }
    }
    /**
     * @return {undefined}
     */
    function local$$67673() {
      if (local$$67458[enabled] === false) {
        return;
      }
      document[removeEventListener](mousemove, local$$67664, false);
      document[removeEventListener](mouseup, local$$67673, false);
      local$$67458[dispatchEvent](local$$67959);
      local$$67556 = local$$67557[_0x34b6[2079]];
    }
    /**
     * @param {?} local$$67970
     * @return {undefined}
     */
    function local$$67969(local$$67970) {
      if (local$$67458[enabled] === false || local$$67458[enableZoom] === false || local$$67556 !== local$$67557[_0x34b6[2079]]) {
        return;
      }
      local$$67970[preventDefault]();
      local$$67970[stopPropagation]();
      /** @type {number} */
      var local$$68003 = 0;
      if (local$$67970[wheelDelta] !== undefined) {
        local$$68003 = local$$67970[wheelDelta];
      } else {
        if (local$$67970[detail] !== undefined) {
          /** @type {number} */
          local$$68003 = -local$$67970[detail];
        }
      }
      if (local$$68003 > 0) {
        local$$67475[_0x34b6[2062]](local$$67507());
      } else {
        if (local$$68003 < 0) {
          local$$67475[_0x34b6[2059]](local$$67507());
        }
      }
      local$$67458[update]();
      local$$67458[dispatchEvent](local$$67680);
      local$$67458[dispatchEvent](local$$67959);
    }
    /**
     * @param {?} local$$68070
     * @return {undefined}
     */
    function local$$68069(local$$68070) {
      if (local$$67458[enabled] === false || local$$67458[_0x34b6[2075]] === false || local$$67458[_0x34b6[2071]] === false) {
        return;
      }
      switch(local$$68070[_0x34b6[2092]]) {
        case local$$67458[_0x34b6[2076]][_0x34b6[2090]]:
          local$$67454(0, local$$67458[_0x34b6[2072]]);
          local$$67458[update]();
          break;
        case local$$67458[_0x34b6[2076]][_0x34b6[2091]]:
          local$$67454(0, -local$$67458[_0x34b6[2072]]);
          local$$67458[update]();
          break;
        case local$$67458[_0x34b6[2076]][LEFT]:
          local$$67454(local$$67458[_0x34b6[2072]], 0);
          local$$67458[update]();
          break;
        case local$$67458[_0x34b6[2076]][RIGHT]:
          local$$67454(-local$$67458[_0x34b6[2072]], 0);
          local$$67458[update]();
          break;
      }
    }
    /**
     * @param {?} local$$68182
     * @return {undefined}
     */
    function local$$68181(local$$68182) {
      if (local$$67458[enabled] === false) {
        return;
      }
      switch(local$$68182[touches][length]) {
        case 1:
          if (local$$67458[_0x34b6[2069]] === false) {
            return;
          }
          local$$67556 = local$$67557[_0x34b6[2093]];
          local$$67563[set](local$$68182[touches][0][_0x34b6[2094]], local$$68182[touches][0][_0x34b6[2095]]);
          break;
        case 2:
          if (local$$67458[enableZoom] === false) {
            return;
          }
          /** @type {number} */
          var local$$68265 = local$$68182[touches][0][_0x34b6[2094]] - local$$68182[touches][1][_0x34b6[2094]];
          /** @type {number} */
          var local$$68284 = local$$68182[touches][0][_0x34b6[2095]] - local$$68182[touches][1][_0x34b6[2095]];
          var local$$68293 = Math[sqrt](local$$68265 * local$$68265 + local$$68284 * local$$68284);
          local$$67829[set](0, local$$68293);
          local$$68301[set](local$$68182[touches][0][_0x34b6[2094]], local$$68182[touches][0][_0x34b6[2095]]);
          local$$68323[set](local$$68182[touches][1][_0x34b6[2094]], local$$68182[touches][1][_0x34b6[2095]]);
          if (local$$67458[_0x34b6[2071]] === false) {
            return;
          }
          local$$67600[set](local$$68182[touches][0][_0x34b6[2094]], local$$68182[touches][0][_0x34b6[2095]]);
          break;
        default:
          local$$67556 = local$$67557[_0x34b6[2079]];
      }
      if (local$$67556 !== local$$67557[_0x34b6[2079]]) {
        local$$67458[dispatchEvent](local$$67680);
      }
    }
    /**
     * @param {?} local$$68401
     * @return {undefined}
     */
    function local$$68400(local$$68401) {
      if (local$$67458[enabled] === false) {
        return;
      }
      local$$68401[preventDefault]();
      local$$68401[stopPropagation]();
      var local$$68436 = local$$67458[domElement] === document ? local$$67458[domElement][body] : local$$67458[domElement];
      switch(local$$68401[touches][length]) {
        case 1:
          if (local$$67458[_0x34b6[2069]] === false) {
            return;
          }
          if (local$$67556 !== local$$67557[_0x34b6[2093]]) {
            return;
          }
          local$$67732[set](local$$68401[touches][0][_0x34b6[2094]], local$$68401[touches][0][_0x34b6[2095]]);
          local$$67744[subVectors](local$$67732, local$$67563);
          local$$67475[_0x34b6[2051]](2 * Math[PI] * local$$67744[x] / local$$68436[clientWidth] * local$$67458[_0x34b6[2070]]);
          local$$67475[_0x34b6[2052]](2 * Math[PI] * local$$67744[y] / local$$68436[clientHeight] * local$$67458[_0x34b6[2070]]);
          local$$67563[copy](local$$67732);
          local$$67458[update]();
          break;
        case 2:
          if (local$$67458[enableZoom] === false) {
            return;
          }
          /** @type {number} */
          var local$$68572 = local$$68401[touches][0][_0x34b6[2094]] - local$$68401[touches][1][_0x34b6[2094]];
          /** @type {number} */
          var local$$68591 = local$$68401[touches][0][_0x34b6[2095]] - local$$68401[touches][1][_0x34b6[2095]];
          var local$$68600 = Math[sqrt](local$$68572 * local$$68572 + local$$68591 * local$$68591);
          var local$$68604 = new THREE.Vector2;
          var local$$68608 = new THREE.Vector2;
          local$$68604[set](local$$68401[touches][0][_0x34b6[2094]], local$$68401[touches][0][_0x34b6[2095]]);
          local$$68608[set](local$$68401[touches][1][_0x34b6[2094]], local$$68401[touches][1][_0x34b6[2095]]);
          local$$68604[subVectors](local$$68604, local$$68301);
          local$$68608[subVectors](local$$68608, local$$68323);
          local$$68301[set](local$$68401[touches][0][_0x34b6[2094]], local$$68401[touches][0][_0x34b6[2095]]);
          local$$68323[set](local$$68401[touches][1][_0x34b6[2094]], local$$68401[touches][1][_0x34b6[2095]]);
          local$$67813[set](0, local$$68600);
          local$$67825[subVectors](local$$67813, local$$67829);
          local$$67879[set](local$$68401[touches][0][_0x34b6[2094]], local$$68401[touches][0][_0x34b6[2095]]);
          local$$67829[copy](local$$67813);
          if (Math[abs](local$$67825[y]) > 2) {
            local$$67556 = local$$67557[_0x34b6[2096]];
            if (local$$67825[y] > 0) {
              local$$67475[_0x34b6[2062]](local$$67507());
            } else {
              if (local$$67825[y] < 0) {
                local$$67475[_0x34b6[2059]](local$$67507());
              }
            }
          } else {
            if (local$$68604[length]() > 2 && local$$68608[length]() > 2) {
              if (local$$67458[_0x34b6[2071]] === false) {
                return;
              }
              local$$67556 = local$$67557[_0x34b6[2097]];
              local$$67891[subVectors](local$$67879, local$$67600);
              local$$67454(local$$67891[x], local$$67891[y]);
            }
          }
          local$$67600[copy](local$$67879);
          local$$67458[update]();
          break;
        default:
          local$$67556 = local$$67557[_0x34b6[2079]];
      }
    }
    /**
     * @return {undefined}
     */
    function local$$68851() {
      if (local$$67458[enabled] === false) {
        return;
      }
      local$$67458[dispatchEvent](local$$67959);
      local$$67556 = local$$67557[_0x34b6[2079]];
    }
    /**
     * @param {?} local$$68875
     * @return {undefined}
     */
    function local$$68874(local$$68875) {
      local$$68875[preventDefault]();
    }
    var local$$67475 = new local$$66411(local$$67451);
    this[domElement] = local$$67452 !== undefined ? local$$67452 : document;
    Object[_0x34b6[2067]](this, _0x34b6[2066], {
      get : function() {
        return local$$67475;
      }
    });
    /**
     * @return {?}
     */
    this[_0x34b6[2049]] = function() {
      return local$$67475[_0x34b6[2049]]();
    };
    /**
     * @return {?}
     */
    this[_0x34b6[2050]] = function() {
      return local$$67475[_0x34b6[2050]]();
    };
    /** @type {boolean} */
    this[enabled] = true;
    this[center] = this[target];
    /** @type {boolean} */
    this[enableZoom] = true;
    /** @type {number} */
    this[_0x34b6[2068]] = 1;
    /** @type {boolean} */
    this[_0x34b6[2069]] = true;
    /** @type {number} */
    this[_0x34b6[2070]] = 1;
    /** @type {boolean} */
    this[_0x34b6[2071]] = true;
    /** @type {number} */
    this[_0x34b6[2072]] = 7;
    /** @type {boolean} */
    this[_0x34b6[2073]] = false;
    /** @type {number} */
    this[_0x34b6[2074]] = 2;
    /** @type {boolean} */
    this[_0x34b6[2075]] = true;
    this[_0x34b6[2076]] = {
      LEFT : 37,
      UP : 38,
      RIGHT : 39,
      BOTTOM : 40
    };
    this[_0x34b6[2077]] = {
      ORBIT : THREE[MOUSE][LEFT],
      ZOOM : THREE[MOUSE][_0x34b6[2078]],
      PAN : THREE[MOUSE][RIGHT]
    };
    var local$$67458 = this;
    var local$$67563 = new THREE.Vector2;
    var local$$67732 = new THREE.Vector2;
    var local$$67744 = new THREE.Vector2;
    var local$$67600 = new THREE.Vector2;
    var local$$67879 = new THREE.Vector2;
    var local$$67891 = new THREE.Vector2;
    var local$$67829 = new THREE.Vector2;
    var local$$67813 = new THREE.Vector2;
    var local$$67825 = new THREE.Vector2;
    var local$$68301 = new THREE.Vector2;
    var local$$68323 = new THREE.Vector2;
    var local$$67557 = {
      NONE : -1,
      ROTATE : 0,
      DOLLY : 1,
      PAN : 2,
      TOUCH_ROTATE : 3,
      TOUCH_DOLLY : 4,
      TOUCH_PAN : 5
    };
    var local$$67556 = local$$67557[_0x34b6[2079]];
    this[_0x34b6[2080]] = this[target][clone]();
    this[_0x34b6[2081]] = this[object][position][clone]();
    this[_0x34b6[2082]] = this[object][_0x34b6[2060]];
    var local$$69121 = {
      type : change
    };
    var local$$67680 = {
      type : start
    };
    var local$$67959 = {
      type : end
    };
    /**
     * @return {undefined}
     */
    this[update] = function() {
      if (this[_0x34b6[2073]] && local$$67556 === local$$67557[_0x34b6[2079]]) {
        local$$67475[_0x34b6[2051]](local$$67489());
      }
      if (local$$67475[update]() === true) {
        this[dispatchEvent](local$$69121);
      }
    };
    /**
     * @return {undefined}
     */
    this[_0x34b6[2083]] = function() {
      local$$67556 = local$$67557[_0x34b6[2079]];
      this[target][copy](this[_0x34b6[2080]]);
      this[object][position][copy](this[_0x34b6[2081]]);
      this[object][_0x34b6[2060]] = this[_0x34b6[2082]];
      this[object][updateProjectionMatrix]();
      this[dispatchEvent](local$$69121);
      this[update]();
    };
    /**
     * @return {undefined}
     */
    this[dispose] = function() {
      this[domElement][removeEventListener](contextmenu, local$$68874, false);
      this[domElement][removeEventListener](mousedown, local$$67520, false);
      this[domElement][removeEventListener](mousewheel, local$$67969, false);
      this[domElement][removeEventListener](MozMousePixelScroll, local$$67969, false);
      this[domElement][removeEventListener](touchstart, local$$68181, false);
      this[domElement][removeEventListener](touchend, local$$68851, false);
      this[domElement][removeEventListener](touchmove, local$$68400, false);
      document[removeEventListener](mousemove, local$$67664, false);
      document[removeEventListener](mouseup, local$$67673, false);
      window[removeEventListener](_0x34b6[2098], local$$68069, false);
    };
    this[domElement][addEventListener](contextmenu, local$$68874, false);
    this[domElement][addEventListener](mousedown, local$$67520, false);
    this[domElement][addEventListener](mousewheel, local$$67969, false);
    this[domElement][addEventListener](MozMousePixelScroll, local$$67969, false);
    this[domElement][addEventListener](touchstart, local$$68181, false);
    this[domElement][addEventListener](touchend, local$$68851, false);
    this[domElement][addEventListener](touchmove, local$$68400, false);
    window[addEventListener](_0x34b6[2098], local$$68069, false);
    this[update]();
  };
  THREE[_0x34b6[2065]][prototype] = Object[create](THREE[EventDispatcher][prototype]);
  THREE[_0x34b6[2065]][prototype][constructor] = THREE[_0x34b6[2065]];
  Object[_0x34b6[2106]](THREE[_0x34b6[2065]][prototype], {
    object : {
      get : function() {
        return this[_0x34b6[2066]][object];
      }
    },
    target : {
      get : function() {
        return this[_0x34b6[2066]][target];
      },
      set : function(local$$69501) {
        console[warn](_0x34b6[2099]);
        this[_0x34b6[2066]][target][copy](local$$69501);
        this[_0x34b6[2066]][_0x34b6[2047]]();
      }
    },
    minDistance : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2039]];
      },
      set : function(local$$69542) {
        this[_0x34b6[2066]][_0x34b6[2039]] = local$$69542;
      }
    },
    maxDistance : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2040]];
      },
      set : function(local$$69565) {
        this[_0x34b6[2066]][_0x34b6[2040]] = local$$69565;
      }
    },
    minZoom : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2041]];
      },
      set : function(local$$69588) {
        this[_0x34b6[2066]][_0x34b6[2041]] = local$$69588;
      }
    },
    maxZoom : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2042]];
      },
      set : function(local$$69611) {
        this[_0x34b6[2066]][_0x34b6[2042]] = local$$69611;
      }
    },
    minPolarAngle : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2043]];
      },
      set : function(local$$69634) {
        this[_0x34b6[2066]][_0x34b6[2043]] = local$$69634;
      }
    },
    maxPolarAngle : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2044]];
      },
      set : function(local$$69657) {
        this[_0x34b6[2066]][_0x34b6[2044]] = local$$69657;
      }
    },
    minAzimuthAngle : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2045]];
      },
      set : function(local$$69680) {
        this[_0x34b6[2066]][_0x34b6[2045]] = local$$69680;
      }
    },
    maxAzimuthAngle : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[2046]];
      },
      set : function(local$$69703) {
        this[_0x34b6[2066]][_0x34b6[2046]] = local$$69703;
      }
    },
    enableDamping : {
      get : function() {
        return this[_0x34b6[2066]][enableDamping];
      },
      set : function(local$$69726) {
        this[_0x34b6[2066]][enableDamping] = local$$69726;
      }
    },
    dampingFactor : {
      get : function() {
        return this[_0x34b6[2066]][dampingFactor];
      },
      set : function(local$$69749) {
        this[_0x34b6[2066]][dampingFactor] = local$$69749;
      }
    },
    noZoom : {
      get : function() {
        console[warn](_0x34b6[2100]);
        return !this[enableZoom];
      },
      set : function(local$$69777) {
        console[warn](_0x34b6[2100]);
        /** @type {boolean} */
        this[enableZoom] = !local$$69777;
      }
    },
    noRotate : {
      get : function() {
        console[warn](_0x34b6[2101]);
        return !this[_0x34b6[2069]];
      },
      set : function(local$$69811) {
        console[warn](_0x34b6[2101]);
        /** @type {boolean} */
        this[_0x34b6[2069]] = !local$$69811;
      }
    },
    noPan : {
      get : function() {
        console[warn](_0x34b6[2102]);
        return !this[_0x34b6[2071]];
      },
      set : function(local$$69845) {
        console[warn](_0x34b6[2102]);
        /** @type {boolean} */
        this[_0x34b6[2071]] = !local$$69845;
      }
    },
    noKeys : {
      get : function() {
        console[warn](_0x34b6[2103]);
        return !this[_0x34b6[2075]];
      },
      set : function(local$$69879) {
        console[warn](_0x34b6[2103]);
        /** @type {boolean} */
        this[_0x34b6[2075]] = !local$$69879;
      }
    },
    staticMoving : {
      get : function() {
        console[warn](_0x34b6[2104]);
        return !this[_0x34b6[2066]][enableDamping];
      },
      set : function(local$$69916) {
        console[warn](_0x34b6[2104]);
        /** @type {boolean} */
        this[_0x34b6[2066]][enableDamping] = !local$$69916;
      }
    },
    dynamicDampingFactor : {
      get : function() {
        console[warn](_0x34b6[2105]);
        return this[_0x34b6[2066]][dampingFactor];
      },
      set : function(local$$69955) {
        console[warn](_0x34b6[2105]);
        this[_0x34b6[2066]][dampingFactor] = local$$69955;
      }
    }
  });
})();
/**
 * @param {?} local$$69985
 * @return {undefined}
 */
LSJTextSprite = function(local$$69985) {
  THREE[Object3D][call](this);
  var local$$69995 = this;
  var local$$69999 = new THREE.Texture;
  local$$69999[minFilter] = THREE[LinearFilter];
  local$$69999[magFilter] = THREE[LinearFilter];
  var local$$70024 = new THREE.SpriteMaterial({
    map : local$$69999,
    useScreenCoordinates : true,
    depthTest : false,
    depthWrite : false
  });
  this[material] = local$$70024;
  this[_0x34b6[2107]] = new THREE.Sprite(local$$70024);
  this[add](this[_0x34b6[2107]]);
  /** @type {number} */
  this[_0x34b6[2108]] = 4;
  this[_0x34b6[2109]] = _0x34b6[2032];
  /** @type {number} */
  this[fontsize] = 28;
  this[_0x34b6[2110]] = {
    r : 0,
    g : 0,
    b : 0,
    a : 1
  };
  this[backgroundColor] = {
    r : 255,
    g : 255,
    b : 255,
    a : 1
  };
  this[_0x34b6[2111]] = {
    r : 255,
    g : 255,
    b : 255,
    a : 1
  };
  this[text] = ;
  this[_0x34b6[2037]](local$$69985);
};
LSJTextSprite[prototype] = new THREE.Object3D;
/**
 * @param {?} local$$70124
 * @return {undefined}
 */
LSJTextSprite[prototype][_0x34b6[2037]] = function(local$$70124) {
  if (this[text] !== local$$70124) {
    this[text] = local$$70124;
    this[update]();
  }
};
/**
 * @return {?}
 */
LSJTextSprite[prototype][_0x34b6[2112]] = function() {
  return this[_0x34b6[2107]];
};
/**
 * @param {?} local$$70167
 * @return {undefined}
 */
LSJTextSprite[prototype][_0x34b6[2113]] = function(local$$70167) {
  this[_0x34b6[2111]] = local$$70167;
  this[update]();
};
/**
 * @param {?} local$$70189
 * @return {undefined}
 */
LSJTextSprite[prototype][_0x34b6[2035]] = function(local$$70189) {
  this[_0x34b6[2110]] = local$$70189;
  this[update]();
};
/**
 * @param {?} local$$70211
 * @return {undefined}
 */
LSJTextSprite[prototype][_0x34b6[2036]] = function(local$$70211) {
  this[backgroundColor] = local$$70211;
  this[update]();
};
/**
 * @return {undefined}
 */
LSJTextSprite[prototype][update] = function() {
  var local$$70240 = document[createElement](canvas);
  var local$$70248 = local$$70240[getContext](2d);
  local$$70248[font] = Bold  + this[fontsize] + px  + this[_0x34b6[2109]];
  var local$$70275 = local$$70248[measureText](this[text]);
  var local$$70280 = local$$70275[width];
  /** @type {number} */
  var local$$70283 = 5;
  var local$$70294 = 2 * local$$70283 + local$$70280 + 2 * this[_0x34b6[2108]];
  /** @type {number} */
  var local$$70307 = this[fontsize] * 1.4 + 2 * this[_0x34b6[2108]];
  local$$70240 = document[createElement](canvas);
  local$$70248 = local$$70240[getContext](2d);
  local$$70248[canvas][width] = local$$70294;
  /** @type {number} */
  local$$70248[canvas][height] = local$$70307;
  local$$70248[font] = Bold  + this[fontsize] + px  + this[_0x34b6[2109]];
  local$$70248[fillStyle] = rgba( + this[backgroundColor][r] + , + this[backgroundColor][g] + , + this[backgroundColor][b] + , + this[backgroundColor][a] + );
  local$$70248[strokeStyle] = rgba( + this[_0x34b6[2110]][r] + , + this[_0x34b6[2110]][g] + , + this[_0x34b6[2110]][b] + , + this[_0x34b6[2110]][a] + );
  local$$70248[lineWidth] = this[_0x34b6[2108]];
  this[_0x34b6[2114]](local$$70248, this[_0x34b6[2108]] / 2, this[_0x34b6[2108]] / 2, local$$70280 + this[_0x34b6[2108]] + 2 * local$$70283, this[fontsize] * 1.4 + this[_0x34b6[2108]], 6);
  local$$70248[strokeStyle] = _0x34b6[2115];
  local$$70248[strokeText](this[text], this[_0x34b6[2108]] + local$$70283, this[fontsize] + this[_0x34b6[2108]]);
  local$$70248[fillStyle] = rgba( + this[_0x34b6[2111]][r] + , + this[_0x34b6[2111]][g] + , + this[_0x34b6[2111]][b] + , + this[_0x34b6[2111]][a] + );
  local$$70248[fillText](this[text], this[_0x34b6[2108]] + local$$70283, this[fontsize] + this[_0x34b6[2108]]);
  var local$$70587 = new THREE.Texture(local$$70240);
  local$$70587[minFilter] = THREE[LinearFilter];
  local$$70587[magFilter] = THREE[LinearFilter];
  /** @type {boolean} */
  local$$70587[needsUpdate] = true;
  this[_0x34b6[2107]][material][map] = local$$70587;
  this[_0x34b6[2107]][scale][set](local$$70294 * .01, local$$70307 * .01, 1);
};
/**
 * @return {?}
 */
LSJTextSprite[prototype][map] = function() {
  return this[_0x34b6[2107]][material][map];
};
/**
 * @param {?} local$$70669
 * @param {(Object|number)} local$$70670
 * @param {(Object|number)} local$$70671
 * @param {!Object} local$$70672
 * @param {!Object} local$$70673
 * @param {!Object} local$$70674
 * @return {undefined}
 */
LSJTextSprite[prototype][_0x34b6[2114]] = function(local$$70669, local$$70670, local$$70671, local$$70672, local$$70673, local$$70674) {
  local$$70669[beginPath]();
  local$$70669[moveTo](local$$70670 + local$$70674, local$$70671);
  local$$70669[_0x34b6[2116]](local$$70670 + local$$70672 - local$$70674, local$$70671);
  local$$70669[_0x34b6[2117]](local$$70670 + local$$70672, local$$70671, local$$70670 + local$$70672, local$$70671 + local$$70674);
  local$$70669[_0x34b6[2116]](local$$70670 + local$$70672, local$$70671 + local$$70673 - local$$70674);
  local$$70669[_0x34b6[2117]](local$$70670 + local$$70672, local$$70671 + local$$70673, local$$70670 + local$$70672 - local$$70674, local$$70671 + local$$70673);
  local$$70669[_0x34b6[2116]](local$$70670 + local$$70674, local$$70671 + local$$70673);
  local$$70669[_0x34b6[2117]](local$$70670, local$$70671 + local$$70673, local$$70670, local$$70671 + local$$70673 - local$$70674);
  local$$70669[_0x34b6[2116]](local$$70670, local$$70671 + local$$70674);
  local$$70669[_0x34b6[2117]](local$$70670, local$$70671, local$$70670 + local$$70674, local$$70671);
  local$$70669[closePath]();
  local$$70669[fill]();
  local$$70669[stroke]();
};
/**
 * @return {undefined}
 */
LSJModelNodeTexture = function() {
  /** @type {number} */
  this[id] = -1;
  this[status] = LSELoadStatus[LS_UNLOAD];
  this[imgUrl] = ;
  /** @type {null} */
  this[texture] = null;
  /** @type {boolean} */
  this[bImgBlobUrl] = false;
  /** @type {boolean} */
  this[_0x34b6[2118]] = false;
};
/**
 * @return {undefined}
 */
LSJModelNodeMaterial = function() {
  /** @type {null} */
  this[material] = null;
  /** @type {!Array} */
  this[_0x34b6[2119]] = [];
};
/**
 * @return {undefined}
 */
LSJModelLODNode = function() {
  this[type] = _0x34b6[2120];
  /** @type {!Array} */
  this[children] = [];
  /** @type {!Array} */
  this[childRanges] = [];
  /** @type {null} */
  this[_0x34b6[2121]] = null;
  /** @type {null} */
  this[parent] = null;
  /** @type {null} */
  this[root] = null;
  this[strDataPath] = ;
  this[meshGroup] = new THREE.Group;
  /** @type {boolean} */
  this[meshGroup][castShadow] = true;
  /** @type {boolean} */
  this[meshGroup][receiveShadow] = true;
  /** @type {boolean} */
  this[bNormalRendered] = false;
  /** @type {boolean} */
  this[bInFrustumTestOk] = false;
  this[bdSphere] = new THREE.Sphere;
  this[bdBox] = new THREE.Box3;
  this[btLoadStatus] = LSELoadStatus[LS_UNLOAD];
  /** @type {number} */
  this[enRangeMode] = 0;
  /** @type {number} */
  this[lastAccessFrame] = 0;
  /** @type {number} */
  this[lastAccessTime] = 0;
  /** @type {boolean} */
  this[bHasGeometry] = false;
  /** @type {!Array} */
  this[arryMaterials] = [];
  /** @type {!Array} */
  this[_0x34b6[2122]] = [];
  /** @type {!Array} */
  this[arryMaterialUsed] = [];
  /** @type {!Array} */
  this[_0x34b6[2123]] = [];
  /** @type {null} */
  this[dataBuffer] = null;
  /** @type {number} */
  this[distToEyeSquare] = 0;
  /** @type {boolean} */
  this[_0x34b6[2124]] = false;
  /** @type {boolean} */
  this[_0x34b6[2125]] = false;
};
/** @type {function(): undefined} */
LSJModelLODNode[prototype][constructor] = LSJModelLODNode;
/**
 * @param {?} local$$71026
 * @return {undefined}
 */
LSJModelLODNode[prototype][setInFrustumTestOk] = function(local$$71026) {
  this[bInFrustumTestOk] = local$$71026;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][isInFrustumTestOk] = function() {
  return this[bInFrustumTestOk];
};
/**
 * @param {?} local$$71058
 * @return {undefined}
 */
LSJModelLODNode[prototype][setLoadStatus] = function(local$$71058) {
  this[btLoadStatus] = local$$71058;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][hasGeometry] = function() {
  return this[bHasGeometry];
};
/**
 * @param {?} local$$71090
 * @return {undefined}
 */
LSJModelLODNode[prototype][setHasGeometry] = function(local$$71090) {
  this[bHasGeometry] = local$$71090;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][getLoadStatus] = function() {
  return this[btLoadStatus];
};
/**
 * @param {?} local$$71122
 * @return {undefined}
 */
LSJModelLODNode[prototype][setLastAccessTime] = function(local$$71122) {
  this[lastAccessTime] = local$$71122;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][getLastAccessTime] = function() {
  return this[lastAccessTime];
};
/**
 * @param {?} local$$71154
 * @return {undefined}
 */
LSJModelLODNode[prototype][setLastAccessFrame] = function(local$$71154) {
  this[lastAccessFrame] = local$$71154;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][getLastAccessFrame] = function() {
  return this[lastAccessFrame];
};
/**
 * @param {?} local$$71186
 * @return {undefined}
 */
LSJModelLODNode[prototype][_0x34b6[2126]] = function(local$$71186) {
  this[_0x34b6[2125]] = local$$71186;
  var local$$71199 = this[children][length];
  /** @type {number} */
  var local$$71202 = 0;
  for (; local$$71202 < local$$71199; local$$71202++) {
    var local$$71211 = this[children][local$$71202];
    local$$71211[_0x34b6[2126]](local$$71186);
  }
};
/**
 * @param {?} local$$71230
 * @return {undefined}
 */
LSJModelLODNode[prototype][addNode] = function(local$$71230) {
  this[children][push](local$$71230);
  local$$71230[_0x34b6[2121]] = this[_0x34b6[2121]];
  local$$71230[root] = this[root];
  local$$71230[parent] = this;
  this[meshGroup][add](local$$71230[meshGroup]);
};
/**
 * @param {?} local$$71282
 * @param {?} local$$71283
 * @param {?} local$$71284
 * @return {?}
 */
LSJModelLODNode[prototype][loadTexture] = function(local$$71282, local$$71283, local$$71284) {
  if (local$$71283[curTexRequestNum] > local$$71283[maxTexRequestNum]) {
    return;
  }
  var local$$71300 = local$$71282[texture];
  if (local$$71300 == null || local$$71300 === undefined) {
    local$$71282[status] = LSELoadStatus[LS_LOADED];
    return null;
  }
  local$$71282[status] = LSELoadStatus[LS_LOADING];
  local$$71283[curTexRequestNum]++;
  var local$$71335 = local$$71282[imgUrl];
  var local$$71354 = local$$71335[substring](local$$71335[lastIndexOf](.), local$$71335[length])[toLowerCase]();
  if (local$$71354 == _0x34b6[2127]) {
    local$$71282[texture] = this[_0x34b6[2128]](local$$71282, local$$71283, local$$71284);
  } else {
    if (local$$71354 == _0x34b6[2129]) {
      local$$71282[texture] = this[_0x34b6[2130]](local$$71282, local$$71283, local$$71284);
    } else {
      /**
       * @param {?} local$$71382
       * @return {undefined}
       */
      var local$$71386 = function(local$$71382) {
      };
      /**
       * @param {?} local$$71388
       * @return {undefined}
       */
      var local$$71422 = function(local$$71388) {
        if (local$$71282[bImgBlobUrl]) {
          window[URL][revokeObjectURL](local$$71282[imgUrl]);
        }
        local$$71282[status] = LSELoadStatus[LS_LOADED];
        local$$71283[curTexRequestNum]--;
      };
      var local$$71437 = THREE[Loader][Handlers][get](local$$71282[imgUrl]);
      var local$$71442 = THREE[DefaultLoadingManager];
      if (local$$71437 !== null) {
        local$$71300 = local$$71437[load](local$$71282[imgUrl], local$$71284);
      } else {
        local$$71437 = new THREE.ImageLoader(local$$71442);
        local$$71437[setCrossOrigin]();
        local$$71437[load](local$$71282[imgUrl], function(local$$71474) {
          local$$71300[image] = local$$71474;
          /** @type {boolean} */
          local$$71300[needsUpdate] = true;
          local$$71300[side] = THREE[FrontSide];
          local$$71300[wrapS] = THREE[RepeatWrapping];
          local$$71300[wrapT] = THREE[RepeatWrapping];
          local$$71300[minFilter] = THREE[_0x34b6[2131]];
          local$$71300[magFilter] = THREE[LinearFilter];
          /** @type {boolean} */
          local$$71300[generateMipmaps] = true;
          if (local$$71282[bImgBlobUrl]) {
            window[URL][revokeObjectURL](local$$71282[imgUrl]);
          }
          local$$71282[imgUrl] = ;
          local$$71282[status] = LSELoadStatus[LS_LOADED];
          local$$71283[curTexRequestNum]--;
        }, local$$71386, local$$71422);
      }
    }
  }
  return local$$71300;
};
/**
 * @param {?} local$$71592
 * @param {?} local$$71593
 * @param {!Function} local$$71594
 * @return {?}
 */
LSJModelLODNode[prototype][_0x34b6[2128]] = function(local$$71592, local$$71593, local$$71594) {
  /**
   * @return {undefined}
   */
  var local$$71599 = function() {
  };
  /**
   * @return {undefined}
   */
  var local$$71634 = function() {
    if (local$$71592[bImgBlobUrl]) {
      window[URL][revokeObjectURL](local$$71592[imgUrl]);
    }
    local$$71592[status] = LSELoadStatus[LS_LOADED];
    local$$71593[curTexRequestNum]--;
  };
  /**
   * @param {?} local$$71636
   * @return {undefined}
   */
  local$$71594 = function(local$$71636) {
    /** @type {boolean} */
    local$$71636[needsUpdate] = true;
    local$$71636[side] = THREE[FrontSide];
    local$$71636[wrapS] = THREE[RepeatWrapping];
    local$$71636[wrapT] = THREE[RepeatWrapping];
    local$$71636[minFilter] = THREE[LinearFilter];
    local$$71636[magFilter] = THREE[LinearFilter];
    /** @type {boolean} */
    local$$71636[generateMipmaps] = false;
    if (local$$71592[bImgBlobUrl]) {
      window[URL][revokeObjectURL](local$$71592[imgUrl]);
    }
    local$$71592[imgUrl] = ;
    local$$71592[status] = LSELoadStatus[LS_LOADED];
    local$$71593[curTexRequestNum]--;
  };
  var local$$71733 = new THREE.PVRLoader;
  var local$$71742 = local$$71733[load](local$$71592[imgUrl], local$$71594, local$$71599, local$$71634);
  return local$$71742;
};
/**
 * @param {?} local$$71755
 * @param {?} local$$71756
 * @param {!Function} local$$71757
 * @return {?}
 */
LSJModelLODNode[prototype][_0x34b6[2130]] = function(local$$71755, local$$71756, local$$71757) {
  /**
   * @return {undefined}
   */
  var local$$71762 = function() {
  };
  /**
   * @return {undefined}
   */
  var local$$71797 = function() {
    if (local$$71755[bImgBlobUrl]) {
      window[URL][revokeObjectURL](local$$71755[imgUrl]);
    }
    local$$71755[status] = LSELoadStatus[LS_LOADED];
    local$$71756[curTexRequestNum]--;
  };
  /**
   * @param {?} local$$71799
   * @return {undefined}
   */
  local$$71757 = function(local$$71799) {
    /** @type {boolean} */
    local$$71799[needsUpdate] = true;
    local$$71799[side] = THREE[FrontSide];
    local$$71799[wrapS] = THREE[_0x34b6[2132]];
    local$$71799[wrapT] = THREE[_0x34b6[2132]];
    local$$71799[minFilter] = THREE[_0x34b6[2133]];
    local$$71799[magFilter] = THREE[_0x34b6[2133]];
    /** @type {number} */
    local$$71799[_0x34b6[2134]] = 4;
    if (local$$71755[bImgBlobUrl]) {
      window[URL][revokeObjectURL](local$$71755[imgUrl]);
    }
    local$$71755[imgUrl] = ;
    local$$71755[status] = LSELoadStatus[LS_LOADED];
    local$$71756[curTexRequestNum]--;
  };
  var local$$71896 = new THREE.DDSLoader;
  var local$$71905 = local$$71896[load](local$$71755[imgUrl], local$$71757, local$$71762, local$$71797);
  return local$$71905;
};
/**
 * @return {undefined}
 */
LSJModelLODNode[prototype][netLoad] = function() {
  if (this[_0x34b6[2121]][curHttpRequestNum] > this[_0x34b6[2121]][maxHttpRequestNum]) {
    return;
  }
  /**
   * @param {?} local$$71936
   * @return {undefined}
   */
  var local$$71955 = function(local$$71936) {
    local$$71938[setLoadStatus](LSELoadStatus.LS_NET_LOADED);
    local$$71938[_0x34b6[2121]][curHttpRequestNum]--;
  };
  this[setLoadStatus](LSELoadStatus.LS_NET_LOADING);
  /** @type {!XMLHttpRequest} */
  var local$$71964 = new XMLHttpRequest;
  local$$71964[open](GET, this[strDataPath], true);
  local$$71964[responseType] = arraybuffer;
  this[_0x34b6[2121]][curHttpRequestNum]++;
  local$$71964[send](null);
  /**
   * @param {?} local$$72001
   * @return {undefined}
   */
  local$$71964[_0x34b6[2135]] = function(local$$72001) {
    if (local$$72001[lengthComputable] && onProgressInfo != undefined) {
      /** @type {number} */
      var local$$72017 = local$$72001[loaded] / local$$72001[total] * 100;
      onProgressInfo(local$$72017);
    }
  };
  var local$$71938 = this;
  /**
   * @return {undefined}
   */
  local$$71964[onreadystatechange] = function() {
    if (local$$71964[readyState] == 4) {
      if (local$$71964[status] == 200) {
        local$$71938[dataBuffer] = local$$71964[response];
      } else {
      }
      local$$71938[setLoadStatus](LSELoadStatus.LS_NET_LOADED);
      local$$71938[_0x34b6[2121]][curHttpRequestNum]--;
      getScene()[curSendNode]--;
    }
  };
};
/**
 * @return {undefined}
 */
LSJModelLODNode[prototype][load] = function() {
  if (this[_0x34b6[2121]][curNodeParseThreadNum] > this[_0x34b6[2121]][maxNodeParseThreadNum]) {
    return;
  }
  if (this[dataBuffer] == null) {
    this[setLoadStatus](LSELoadStatus.LS_LOADED);
    return;
  }
  this[setLoadStatus](LSELoadStatus.LS_LOADING);
  var local$$72129 = this;
  this[_0x34b6[2121]][curNodeParseThreadNum]++;
  /** @type {!Worker} */
  var local$$72142 = new Worker(_0x34b6[2136]);
  /**
   * @param {?} local$$72147
   * @return {undefined}
   */
  local$$72142[onmessage] = function(local$$72147) {
    var local$$72152 = local$$72147[data];
    if (local$$72152 != null && local$$72152 != undefined) {
      var local$$72158;
      {
        var local$$72163 = local$$72129[strDataPath];
        local$$72158 = local$$72163[substr](0, local$$72163[lastIndexOf](/) + 1);
      }
      var local$$72187 = local$$72152[_0x34b6[2137]][length];
      /** @type {number} */
      var local$$72190 = 0;
      for (; local$$72190 < local$$72187; local$$72190++) {
        var local$$72196 = new LSJModelNodeTexture;
        var local$$72202 = local$$72152[_0x34b6[2137]][local$$72190];
        local$$72196[_0x34b6[2118]] = local$$72202[_0x34b6[2118]];
        if (local$$72202[imgUrl] != ) {
          if (local$$72202[bUrl]) {
            local$$72196[imgUrl] = LSJUtility[getAbsolutePath](local$$72158, local$$72202[imgUrl]);
          } else {
            local$$72196[imgUrl] = local$$72202[imgUrl];
          }
        } else {
          if (local$$72202[imgBlob] != null) {
            local$$72196[imgUrl] = window[URL][createObjectURL](local$$72202[imgBlob]);
            /** @type {null} */
            local$$72202[imgBlob] = null;
            /** @type {boolean} */
            local$$72196[bImgBlobUrl] = true;
          }
        }
        if (local$$72196[imgUrl] ==  || local$$72196[imgUrl] === undefined) {
          local$$72196[status] = LSELoadStatus[LS_LOADED];
        }
        local$$72129[_0x34b6[2122]][push](local$$72196);
      }
      var local$$72321 = local$$72152[arryMaterials][length];
      /** @type {number} */
      local$$72190 = 0;
      for (; local$$72190 < local$$72321; local$$72190++) {
        var local$$72333 = local$$72152[arryMaterials][local$$72190];
        var local$$72336 = new LSJModelNodeMaterial;
        var local$$72340 = new THREE.MeshPhongMaterial;
        local$$72336[material] = local$$72340;
        if (local$$72333[_0x34b6[2138]][0] != 0 || local$$72333[_0x34b6[2138]][1] != 0 || local$$72333[_0x34b6[2138]][2] != 0) {
          local$$72340[color] = (new THREE.Color)[setRGB](local$$72333[_0x34b6[2138]][0] / 255, local$$72333[_0x34b6[2138]][1] / 255, local$$72333[_0x34b6[2138]][2] / 255);
        }
        local$$72340[specular] = (new THREE.Color)[setRGB](local$$72333[specular][0] / 255, local$$72333[specular][1] / 255, local$$72333[specular][2] / 255);
        local$$72340[_0x34b6[2139]] = (new THREE.Color)[setRGB](local$$72333[_0x34b6[2139]][0] / 255, local$$72333[_0x34b6[2139]][1] / 255, local$$72333[_0x34b6[2139]][2] / 255);
        local$$72340[shininess] = local$$72333[shininess];
        if (local$$72333[_0x34b6[2138]][3] < 255) {
          /** @type {number} */
          local$$72340[opacity] = local$$72333[_0x34b6[2138]][3] / 255;
          /** @type {boolean} */
          local$$72340[transparent] = true;
        }
        var local$$72511 = local$$72333[_0x34b6[2119]][length];
        /** @type {number} */
        var local$$72514 = 0;
        for (; local$$72514 < local$$72511; local$$72514++) {
          local$$72336[_0x34b6[2119]][push](local$$72333[_0x34b6[2119]][local$$72514]);
        }
        local$$72129[arryMaterials][push](local$$72336);
      }
      local$$72129[parse](local$$72152, local$$72129[_0x34b6[2122]], local$$72129[arryMaterials], local$$72158);
    }
    /** @type {null} */
    local$$72152 = null;
    /** @type {null} */
    local$$72147[data] = null;
    /** @type {null} */
    local$$72129[dataBuffer] = null;
    local$$72129[setLoadStatus](LSELoadStatus.LS_LOADED);
    var local$$72582 = new THREE.Matrix4;
    var local$$72586 = new THREE.Quaternion;
    var local$$72590 = new THREE.Euler;
    local$$72590[order] = XYZ;
    local$$72590[x] = local$$72129[root][_0x34b6[2121]][_0x34b6[2140]]()[x];
    local$$72590[y] = local$$72129[root][_0x34b6[2121]][_0x34b6[2140]]()[y];
    local$$72590[z] = local$$72129[root][_0x34b6[2121]][_0x34b6[2140]]()[z];
    local$$72586[setFromEuler](local$$72590);
    local$$72582[_0x34b6[2142]](local$$72129[root][_0x34b6[2121]][getPosition](), local$$72586, local$$72129[root][_0x34b6[2121]][_0x34b6[2141]]());
    local$$72129[root][bdSphere][applyMatrix4](local$$72582);
    local$$72129[root][bdBox][applyMatrix4](local$$72582);
    /** @type {boolean} */
    local$$72129[_0x34b6[2125]] = true;
    local$$72129[_0x34b6[2121]][curNodeParseThreadNum]--;
  };
  /**
   * @param {?} local$$72726
   * @return {undefined}
   */
  local$$72142[onerror] = function(local$$72726) {
    console[log](Error: + local$$72726[message]);
    /** @type {null} */
    local$$72129[dataBuffer] = null;
    local$$72129[setLoadStatus](LSELoadStatus.LS_LOADED);
    local$$72129[_0x34b6[2121]][curNodeParseThreadNum]--;
  };
  local$$72142[postMessage](this[dataBuffer]);
};
/**
 * @param {!Array} local$$72781
 * @param {?} local$$72782
 * @param {?} local$$72783
 * @param {?} local$$72784
 * @return {undefined}
 */
LSJModelLODNode[prototype][parse] = function(local$$72781, local$$72782, local$$72783, local$$72784) {
  if (local$$72781 == null || local$$72781 === undefined) {
    return;
  }
  /** @type {number} */
  var local$$72795 = 0;
  var local$$72803 = local$$72781[children][length];
  /** @type {number} */
  local$$72795 = 0;
  for (; local$$72795 < local$$72803; local$$72795++) {
    var local$$72812 = new LSJModelLODNode;
    this[addNode](local$$72812);
    local$$72812[parse](local$$72781[children][local$$72795], local$$72782, local$$72783, local$$72784);
  }
  this[enRangeMode] = local$$72781[enRangeMode];
  if (local$$72781[childRanges][length] > 0) {
    /** @type {number} */
    local$$72803 = local$$72781[childRanges][length] / 2;
    /** @type {number} */
    local$$72795 = 0;
    for (; local$$72795 < local$$72803; local$$72795++) {
      var local$$72865 = new THREE.Vector2;
      local$$72865[x] = local$$72781[childRanges][2 * local$$72795];
      local$$72865[y] = local$$72781[childRanges][2 * local$$72795 + 1];
      this[childRanges][push](local$$72865);
    }
  }
  if (this[strDataPath] ==  || this[strDataPath] === undefined) {
    if (local$$72781[strDataPath] != ) {
      this[strDataPath] = local$$72784 + local$$72781[strDataPath];
    }
  }
  if (local$$72781[bdSphere][length] > 0) {
    this[bdSphere] = new THREE.Sphere;
    var local$$72952 = new THREE.Vector3;
    local$$72952[set](local$$72781[bdSphere][0], local$$72781[bdSphere][1], local$$72781[bdSphere][2]);
    if (local$$72952, local$$72781[bdSphere][3] > 1.7E38) {
      local$$72952;
      /** @type {number} */
      local$$72781[bdSphere][3] = 0;
    }
    this[bdSphere][set](local$$72952, local$$72781[bdSphere][3]);
    LSJMath[expandSphere](this[_0x34b6[2121]][_0x34b6[2143]][bdSphere], this[bdSphere]);
  }
  if (local$$72781[bdBox][length] > 0) {
    this[bdSphere] = new THREE.Sphere;
    var local$$73044 = new THREE.Vector3;
    local$$73044[set](local$$72781[bdBox][0], local$$72781[bdBox][1], local$$72781[bdBox][2]);
    var local$$73068 = new THREE.Vector3;
    local$$73068[set](local$$72781[bdBox][3], local$$72781[bdBox][4], local$$72781[bdBox][5]);
    this[bdBox][set](local$$73044, local$$73068);
    this[_0x34b6[2121]][_0x34b6[2143]][bdBox][_0x34b6[2144]](this[bdBox]);
  }
  var local$$73124 = local$$72781[nodeMeshes][length];
  /** @type {number} */
  var local$$73127 = 0;
  for (; local$$73127 < local$$73124; local$$73127++) {
    var local$$73136 = local$$72781[nodeMeshes][local$$73127];
    if (local$$73136[verts] != null) {
      var local$$73145 = new THREE.BufferGeometry;
      if (local$$73136[indices] != null) {
        local$$73145[setIndex](new THREE.BufferAttribute(local$$73136[indices], 1));
      }
      if (local$$73136[verts] != null) {
        local$$73145[addAttribute](position, new THREE.BufferAttribute(local$$73136[verts], 3));
      }
      if (local$$73136[normals] != null) {
        local$$73145[addAttribute](normal, new THREE.BufferAttribute(local$$73136[normals], 3));
      }
      if (local$$73136[colors] != null) {
        local$$73145[addAttribute](color, new THREE.BufferAttribute(local$$73136[colors], local$$73136[colorPerNum]));
      }
      var local$$73237 = local$$73136[uvs][length];
      /** @type {number} */
      k = 0;
      for (; k < local$$73237; k++) {
        if (local$$73136[uvs][k] != null && local$$73136[uvs][k] != undefined) {
          local$$73145[addAttribute](uv, new THREE.BufferAttribute(local$$73136[uvs][k], 2));
        }
      }
      var local$$73282 = local$$73136[_0x34b6[2145]][length];
      /** @type {number} */
      var local$$73285 = -1;
      if (local$$73282 > 0) {
        local$$73145[setIndex](new THREE.BufferAttribute(local$$73136[_0x34b6[2145]][0][indices], 1));
        local$$73285 = local$$73136[_0x34b6[2145]][0][matIndex];
      }
      if (local$$73285 < 0 || local$$73285 >= local$$72783[length]) {
        local$$73285 = local$$73136[matIndex];
      }
      /** @type {null} */
      var local$$73334 = null;
      /** @type {null} */
      var local$$73337 = null;
      if (local$$73285 >= 0 && local$$73285 < local$$72783[length]) {
        local$$73334 = local$$72783[local$$73136[matIndex]];
        local$$73337 = local$$73334[material];
      }
      if (local$$73337 == null) {
        local$$73337 = lmModelDefaultMat;
      }
      var local$$73369 = new THREE.Mesh(local$$73145, local$$73337);
      if (local$$73136[colors] != null && local$$73337 != null) {
        local$$73337[vertexColors] = THREE[VertexColors];
      }
      /** @type {boolean} */
      local$$73369[castShadow] = true;
      /** @type {boolean} */
      local$$73369[receiveShadow] = true;
      /** @type {number} */
      var local$$73403 = 0;
      if (local$$73334 != null) {
        this[arryMaterialUsed][push](local$$73334);
        local$$73403 = local$$73334[_0x34b6[2119]][length];
      }
      /** @type {number} */
      var local$$73427 = 0;
      for (; local$$73427 < local$$73403; local$$73427++) {
        var local$$73439 = local$$73334[_0x34b6[2119]][local$$73427][_0x34b6[2146]];
        if (local$$73439 >= 0 && local$$73439 < local$$72782[length]) {
          this[_0x34b6[2123]][push](local$$72782[local$$73439]);
          if (local$$73334[material][map] == null || local$$73334[material][map] === undefined) {
            if (local$$72782[local$$73439][texture] == null || local$$72782[local$$73439][texture] === undefined) {
              local$$72782[local$$73439][texture] = new THREE.Texture;
            }
            local$$73334[material][map] = local$$72782[local$$73439][texture];
            if (local$$72782[local$$73439][_0x34b6[2118]]) {
              /** @type {boolean} */
              local$$73334[material][transparent] = true;
              /** @type {number} */
              local$$73334[material][alphaTest] = .01;
            }
          }
        }
      }
      this[meshGroup][add](local$$73369);
      this[setHasGeometry](true);
    }
  }
  if (this[strDataPath] == ) {
    this[btLoadStatus] = LSELoadStatus[LS_LOADED];
  }
};
/**
 * @param {?} local$$73584
 * @return {?}
 */
LSJModelLODNode[prototype][checkInFrustum] = function(local$$73584) {
  this[setInFrustumTestOk](false);
  var local$$73599 = this[_0x34b6[2121]][getFrustum]();
  if (!this[bdSphere][empty]()) {
    if (!local$$73599[intersectsSphere](this[bdSphere])) {
      return false;
    }
  }
  this[setInFrustumTestOk](true);
  return true;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][isGrandchildrenSafeDel] = function() {
  if (this[getLoadStatus]() != LSELoadStatus[LS_UNLOAD] && this[getLoadStatus]() != LSELoadStatus[LS_NET_LOADED] && this[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
    return false;
  }
  if (this[hasLoadingMaterial]()) {
    return false;
  }
  /** @type {number} */
  var local$$73684 = 0;
  var local$$73692 = this[children][length];
  for (; local$$73684 < local$$73692; local$$73684++) {
    if (!this[children][local$$73684][isGrandchildrenSafeDel]()) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][_0x34b6[2147]] = function() {
  /** @type {number} */
  var local$$73727 = 0;
  var local$$73735 = this[_0x34b6[2123]][length];
  for (; local$$73727 < local$$73735; local$$73727++) {
    if (this[_0x34b6[2123]][local$$73727][status] != LSELoadStatus[LS_LOADED]) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][_0x34b6[2148]] = function() {
  /** @type {number} */
  var local$$73772 = 0;
  var local$$73780 = this[_0x34b6[2123]][length];
  for (; local$$73772 < local$$73780; local$$73772++) {
    if (this[_0x34b6[2123]][local$$73772][status] != LSELoadStatus[LS_UNLOAD] && this[_0x34b6[2123]][local$$73772][status] != LSELoadStatus[LS_LOADED]) {
      return true;
    }
  }
  return false;
};
/**
 * @return {?}
 */
LSJModelLODNode[prototype][calcNodeCount] = function() {
  /** @type {number} */
  var local$$73829 = 0;
  if (this[hasGeometry]()) {
    /** @type {number} */
    local$$73829 = local$$73829 + 1;
  }
  /** @type {number} */
  var local$$73843 = 0;
  var local$$73851 = this[children][length];
  for (; local$$73843 < local$$73851; local$$73843++) {
    local$$73829 = local$$73829 + this[children][local$$73843][calcNodeCount]();
  }
  return local$$73829;
};
/**
 * @return {undefined}
 */
LSJModelLODNode[prototype][unloadChildren] = function() {
  /** @type {number} */
  var local$$73883 = 0;
  /** @type {number} */
  var local$$73886 = 0;
  var local$$73894 = this[children][length];
  /** @type {number} */
  local$$73886 = 0;
  for (; local$$73886 < local$$73894; local$$73886++) {
    this[children][local$$73886][unloadChildren]();
  }
  this[children][splice](0, local$$73894);
  this[childRanges][splice](0, this[childRanges][length]);
  this[arryMaterialUsed][splice](0, this[arryMaterialUsed][length]);
  this[_0x34b6[2123]][splice](0, this[_0x34b6[2123]][length]);
  /** @type {number} */
  var local$$73979 = this[meshGroup][children][length] - 1;
  for (; local$$73979 >= 0; local$$73979--) {
    var local$$73992 = this[meshGroup][children][local$$73979];
    this[meshGroup][remove](local$$73992);
    if (local$$73992 != null && local$$73992 instanceof THREE[Mesh]) {
      if (local$$73992[geometry]) {
        local$$73992[geometry][dispose]();
      }
      /** @type {null} */
      local$$73992[material] = null;
      /** @type {null} */
      local$$73992[geometry] = null;
      this[_0x34b6[2121]][addReleaseCount](1);
    }
    /** @type {null} */
    local$$73992 = null;
  }
  local$$73894 = this[arryMaterials][length];
  /** @type {number} */
  local$$73886 = 0;
  for (; local$$73886 < local$$73894; local$$73886++) {
    var local$$74071 = this[arryMaterials][local$$73886];
    if (local$$74071[material] != null && local$$74071[material] != undefined) {
      local$$74071[material][dispose]();
      /** @type {null} */
      local$$74071[material][map] = null;
      /** @type {null} */
      local$$74071[material] = null;
    }
  }
  this[arryMaterials][splice](0, local$$73894);
  local$$73894 = this[_0x34b6[2122]][length];
  /** @type {number} */
  local$$73886 = 0;
  for (; local$$73886 < local$$73894; local$$73886++) {
    var local$$74138 = this[_0x34b6[2122]][local$$73886];
    if (local$$74138[texture] != null && local$$74138[texture] != undefined) {
      local$$74138[texture][dispose]();
      /** @type {null} */
      local$$74138[texture][image] = null;
      /** @type {null} */
      local$$74138[texture] = null;
    }
  }
  this[_0x34b6[2122]][splice](0, local$$73894);
  /** @type {boolean} */
  this[bHasGeometry] = false;
  /** @type {null} */
  this[dataBuffer] = null;
  this[setLoadStatus](LSELoadStatus.LS_UNLOAD);
};
/**
 * @param {?} local$$74215
 * @return {?}
 */
LSJModelLODNode[prototype][checkAllGroupChildLoaded] = function(local$$74215) {
  /** @type {number} */
  var local$$74218 = 0;
  var local$$74226 = this[children][length];
  for (; local$$74218 < local$$74226; local$$74218++) {
    var local$$74235 = this[children][local$$74218];
    if (local$$74235 != null) {
      if (local$$74235[checkInFrustum](local$$74215) && local$$74235[children][length] > 1) {
        local$$74235[setInFrustumTestOk](true);
        var local$$74263 = local$$74235[children][0];
        if (local$$74263) {
          if (local$$74263[strDataPath] != ) {
            if (local$$74263[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
};
/**
 * @param {?} local$$74306
 * @return {?}
 */
LSJModelLODNode[prototype][update] = function(local$$74306) {
  /** @type {boolean} */
  this[meshGroup][visible] = false;
  var local$$74326 = this[meshGroup][children][length];
  /** @type {number} */
  var local$$74329 = 0;
  /** @type {number} */
  local$$74329 = 0;
  for (; local$$74329 < local$$74326; local$$74329++) {
    /** @type {boolean} */
    this[meshGroup][children][local$$74329][visible] = false;
  }
  /** @type {boolean} */
  this[bNormalRendered] = false;
  {
    var local$$74361 = new THREE.Matrix4;
    var local$$74365 = new THREE.Quaternion;
    var local$$74369 = new THREE.Euler;
    local$$74369[order] = XYZ;
    local$$74369[x] = this[root][_0x34b6[2121]][_0x34b6[2140]]()[x];
    local$$74369[y] = this[root][_0x34b6[2121]][_0x34b6[2140]]()[y];
    local$$74369[z] = this[root][_0x34b6[2121]][_0x34b6[2140]]()[z];
    local$$74365[setFromEuler](local$$74369);
    local$$74361[_0x34b6[2142]](this[root][_0x34b6[2121]][getPosition](), local$$74365, this[root][_0x34b6[2121]][_0x34b6[2141]]());
    local$$74326 = this[meshGroup][children][length];
    /** @type {number} */
    local$$74329 = 0;
    for (; local$$74329 < local$$74326; local$$74329++) {
      var local$$74486 = this[meshGroup][children][local$$74329];
      if (local$$74486 && local$$74486 instanceof THREE[Mesh]) {
        local$$74486[scale][x] = this[root][_0x34b6[2121]][scale][x];
        local$$74486[scale][y] = this[root][_0x34b6[2121]][scale][y];
        local$$74486[scale][z] = this[root][_0x34b6[2121]][scale][z];
        local$$74486[position][x] = this[root][_0x34b6[2121]][getPosition]()[x];
        local$$74486[position][y] = this[root][_0x34b6[2121]][getPosition]()[y];
        local$$74486[position][z] = this[root][_0x34b6[2121]][getPosition]()[z];
        local$$74486[rotation][x] = this[root][_0x34b6[2121]][rotate][x];
        local$$74486[rotation][y] = this[root][_0x34b6[2121]][rotate][y];
        local$$74486[rotation][z] = this[root][_0x34b6[2121]][rotate][z];
        /** @type {boolean} */
        local$$74486[_0x34b6[2125]] = true;
        this[bdSphere][applyMatrix4](local$$74361);
        this[bdBox][applyMatrix4](local$$74361);
      }
    }
    /** @type {boolean} */
    this[_0x34b6[2125]] = false;
  }
  if (!this[checkInFrustum](local$$74306)) {
    /** @type {boolean} */
    this[meshGroup][visible] = false;
    return false;
  }
  this[setLastAccessTime](this[_0x34b6[2121]][getLastAccessTime]());
  this[setLastAccessFrame](this[_0x34b6[2121]][getLastAccessFrame]());
  if (this[strDataPath] != ) {
    if (this[getLoadStatus]() == LSELoadStatus[LS_UNLOAD]) {
      getScene()[curSendNode]++;
      this[netLoad]();
    }
    if (this[getLoadStatus]() == LSELoadStatus[LS_NET_LOADED]) {
      this[load]();
    }
    if (this[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
      this[_0x34b6[2121]][curLoadingNode]++;
      return false;
    }
  }
  /** @type {number} */
  var local$$74821 = 0;
  if (this[childRanges][length] > 0) {
    if (this[enRangeMode] == LSERangeMode[RM_DISTANCE_FROM_EYE_POINT]) {
      if (!this[bdSphere][empty]()) {
        local$$74821 = LSJMath[computeDistFromEye](this[bdSphere][center], this[_0x34b6[2121]][getModelViewMatrix]());
      }
    } else {
      if (this[enRangeMode] == LSERangeMode[RM_PIXEL_SIZE_ON_SCREEN]) {
        if (!this[bdSphere][empty]()) {
          local$$74821 = LSJMath[computeSpherePixelSize](this[bdSphere], this[_0x34b6[2121]][getPixelSizeVector]());
        }
      }
    }
  }
  /** @type {boolean} */
  var local$$74909 = true;
  /** @type {number} */
  var local$$74912 = 0;
  local$$74326 = this[children][length];
  /** @type {number} */
  local$$74329 = 0;
  for (; local$$74329 < local$$74326; local$$74329++) {
    var local$$74932 = this[children][local$$74329];
    if (local$$74912 < this[childRanges][length]) {
      var local$$74945 = this[childRanges][local$$74912];
      if (local$$74932 && local$$74821 >= local$$74945[x] && local$$74821 < local$$74945[y]) {
        if (local$$74932[update](local$$74306)) {
          /** @type {boolean} */
          this[bNormalRendered] = true;
        } else {
          if (local$$74932[isInFrustumTestOk]()) {
            /** @type {number} */
            var local$$74974 = local$$74329 - 1;
            for (; local$$74974 >= 0; local$$74974--) {
              if (this[children][local$$74974][update](local$$74306)) {
                /** @type {boolean} */
                this[bNormalRendered] = true;
                break;
              }
            }
          }
        }
      }
      local$$74912++;
    } else {
      if (local$$74932 && local$$74909) {
        if (local$$74932[update](local$$74306)) {
          /** @type {boolean} */
          this[bNormalRendered] = true;
        }
      }
    }
  }
  /** @type {boolean} */
  this[meshGroup][visible] = true;
  /** @type {boolean} */
  var local$$75039 = false;
  var local$$75045 = this[_0x34b6[2147]]();
  if (!this[bNormalRendered] && this[_0x34b6[2121]][curTexRequestNum] < this[_0x34b6[2121]][maxTexRequestNum]) {
    local$$74326 = this[_0x34b6[2123]][length];
    /** @type {number} */
    local$$74329 = 0;
    for (; local$$74329 < local$$74326; local$$74329++) {
      var local$$75083 = this[_0x34b6[2123]][local$$74329];
      if (local$$75083[status] == LSELoadStatus[LS_UNLOAD]) {
        this[loadTexture](local$$75083, this[_0x34b6[2121]]);
      }
    }
  }
  if (local$$75045 && !this[_0x34b6[2124]] || this[root][_0x34b6[2121]][_0x34b6[2149]]) {
    /** @type {number} */
    local$$74329 = 0;
    for (; local$$74329 < this[arryMaterialUsed][length]; local$$74329++) {
      var local$$75138 = this[arryMaterialUsed][local$$74329];
      local$$75083 = this[_0x34b6[2123]][local$$74329];
      if (local$$75083 != undefined) {
        local$$75138[material][color] = (new THREE.Color)[setRGB](1, 1, 1);
      }
      if (this[root][_0x34b6[2121]][_0x34b6[2150]]) {
        local$$75138[material][color] = (new THREE.Color)[setRGB](.6, .6, 1);
      }
    }
    /** @type {boolean} */
    this[_0x34b6[2124]] = true;
  }
  local$$74326 = this[meshGroup][children][length];
  /** @type {number} */
  local$$74329 = 0;
  for (; local$$74329 < local$$74326; local$$74329++) {
    local$$74486 = this[meshGroup][children][local$$74329];
    if (local$$74486 && local$$74486 instanceof THREE[Mesh]) {
      if (!this[bNormalRendered] && local$$75045) {
        /** @type {boolean} */
        local$$74486[visible] = true;
        /** @type {boolean} */
        local$$75039 = true;
      } else {
        /** @type {boolean} */
        local$$74486[visible] = false;
      }
    }
  }
  if (!local$$75039) {
    this[meshGroup][visible] = this[bNormalRendered];
    return this[bNormalRendered];
  }
  /** @type {boolean} */
  this[bNormalRendered] = true;
  return true;
};
/**
 * @return {undefined}
 */
LSJGeoModelLOD = function() {
  LSJGeometry[call](this);
  this[type] = GeoModelLOD;
  this[_0x34b6[2151]] = ;
  this[position] = new THREE.Vector3(0, 0, 0);
  this[rotate] = new THREE.Vector3(0, 0, 0);
  this[scale] = new THREE.Vector3(1, 1, 1);
  /** @type {number} */
  this[nodeCount] = 0;
  /** @type {number} */
  this[maxNodeCount] = 200;
  this[_0x34b6[2143]] = new LSJModelLODNode;
  this[_0x34b6[2143]][_0x34b6[2121]] = this;
  this[_0x34b6[2143]][root] = this[_0x34b6[2143]];
  this[meshGroup][add](this[_0x34b6[2143]][meshGroup]);
  this[meshGroup][Owner] = this;
  this[frustum] = new THREE.Frustum;
  this[viewPort] = new THREE.Vector4;
  this[matLocal] = new THREE.Matrix4;
  this[matLocalInvert] = new THREE.Matrix4;
  this[matModelView] = new THREE.Matrix4;
  this[matVPW] = new THREE.Matrix4;
  this[pixelSizeVector] = new THREE.Vector4;
  /** @type {number} */
  this[lastAccessFrame] = 0;
  /** @type {number} */
  this[lastAccessTime] = 0;
  /** @type {number} */
  this[maxHttpRequestNum] = 2;
  /** @type {number} */
  this[curHttpRequestNum] = 0;
  /** @type {number} */
  this[maxTexRequestNum] = 2;
  /** @type {number} */
  this[curTexRequestNum] = 0;
  /** @type {number} */
  this[maxNodeParseThreadNum] = 2;
  /** @type {number} */
  this[curNodeParseThreadNum] = 0;
  /** @type {number} */
  this[curLoadingNode] = 0;
  /** @type {boolean} */
  this[_0x34b6[2125]] = false;
};
LSJGeoModelLOD[prototype] = Object[create](LSJGeometry[prototype]);
/** @type {function(): undefined} */
LSJGeoModelLOD[prototype][constructor] = LSJGeoModelLOD;
/**
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][dispose] = function() {
  this[_0x34b6[2143]][unloadChildren]();
};
/**
 * @param {?} local$$75567
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][_0x34b6[2152]] = function(local$$75567) {
  if (this[_0x34b6[2151]] != local$$75567) {
    this[dispose]();
    var local$$75579 = new LSJModelLODNode;
    local$$75579[strDataPath] = local$$75567;
    this[_0x34b6[2143]][addNode](local$$75579);
    this[_0x34b6[2151]] = local$$75567;
  }
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][_0x34b6[2153]] = function() {
  return this[_0x34b6[2151]];
};
/**
 * @param {?} local$$75626
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][setName] = function(local$$75626) {
  this[name] = local$$75626;
};
/**
 * @param {?} local$$75643
 * @param {?} local$$75644
 * @param {?} local$$75645
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][setPosition] = function(local$$75643, local$$75644, local$$75645) {
  this[position][x] = local$$75643;
  this[position][y] = local$$75644;
  this[position][z] = local$$75645;
  /** @type {boolean} */
  this[_0x34b6[2125]] = true;
  this[_0x34b6[2143]][_0x34b6[2126]](true);
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getPosition] = function() {
  return this[position];
};
/**
 * @param {?} local$$75711
 * @param {?} local$$75712
 * @param {?} local$$75713
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][_0x34b6[2154]] = function(local$$75711, local$$75712, local$$75713) {
  this[rotate][x] = local$$75711;
  this[rotate][y] = local$$75712;
  this[rotate][z] = local$$75713;
  /** @type {boolean} */
  this[_0x34b6[2125]] = true;
  this[_0x34b6[2143]][_0x34b6[2126]](true);
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][_0x34b6[2140]] = function() {
  return this[rotate];
};
/**
 * @param {?} local$$75779
 * @param {?} local$$75780
 * @param {?} local$$75781
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][_0x34b6[2155]] = function(local$$75779, local$$75780, local$$75781) {
  this[scale][x] = local$$75779;
  this[scale][y] = local$$75780;
  this[scale][z] = local$$75781;
  /** @type {boolean} */
  this[_0x34b6[2125]] = true;
  this[_0x34b6[2143]][_0x34b6[2126]](true);
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][_0x34b6[2141]] = function() {
  return this[scale];
};
/**
 * @param {?} local$$75847
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][_0x34b6[2156]] = function(local$$75847) {
  if (this[_0x34b6[2150]] != local$$75847) {
    this[_0x34b6[2150]] = local$$75847;
    /** @type {boolean} */
    this[_0x34b6[2149]] = true;
  }
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][_0x34b6[2157]] = function() {
  return this[_0x34b6[2143]][bdBox];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getBoundingSphere] = function() {
  return this[_0x34b6[2143]][bdSphere];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getPixelSizeVector] = function() {
  return this[pixelSizeVector];
};
/**
 * @param {?} local$$75927
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][setPixelSizeVector] = function(local$$75927) {
  this[pixelSizeVector] = local$$75927;
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getModelViewMatrix] = function() {
  return this[modelViewMatrix];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getFrustum] = function() {
  return this[frustum];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getViewport] = function() {
  return this[viewPort];
};
/**
 * @param {?} local$$75989
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][setViewport] = function(local$$75989) {
  this[viewPort] = local$$75989;
};
/**
 * @param {?} local$$76006
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][setLastAccessTime] = function(local$$76006) {
  this[lastAccessTime] = local$$76006;
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getLastAccessTime] = function() {
  return this[lastAccessTime];
};
/**
 * @param {?} local$$76038
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][setLastAccessFrame] = function(local$$76038) {
  this[lastAccessFrame] = local$$76038;
};
/**
 * @return {?}
 */
LSJGeoModelLOD[prototype][getLastAccessFrame] = function() {
  return this[lastAccessFrame];
};
/**
 * @param {?} local$$76070
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][addReleaseCount] = function(local$$76070) {
  this[nodeCount] -= local$$76070;
};
/**
 * @param {?} local$$76087
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][addNodeCount] = function(local$$76087) {
  this[nodeCount] += local$$76087;
};
/**
 * @param {!Object} local$$76104
 * @return {?}
 */
LSJGeoModelLOD[prototype][cleanRedundantNodes] = function(local$$76104) {
  if (this[nodeCount] < this[maxNodeCount]) {
    return false;
  }
  if (local$$76104 != null) {
    if (local$$76104[getLoadStatus]() != LSELoadStatus[LS_LOADED]) {
      return false;
    }
    /** @type {number} */
    var local$$76134 = 0;
    var local$$76142 = local$$76104[children][length];
    for (; local$$76134 < local$$76142; local$$76134++) {
      this[cleanRedundantNodes](local$$76104[children][local$$76134]);
    }
    if (this[nodeCount] < this[maxNodeCount]) {
      return false;
    }
    if (local$$76104[strDataPath] == ) {
      return false;
    }
    /** @type {number} */
    var local$$76191 = this[getLastAccessFrame]() - local$$76104[getLastAccessFrame]();
    /** @type {number} */
    var local$$76202 = this[getLastAccessTime]() - local$$76104[getLastAccessTime]();
    if (local$$76191 < 5 || local$$76202 < 100) {
      return false;
    }
    if (local$$76104[isGrandchildrenSafeDel]()) {
      local$$76104[unloadChildren]();
      return true;
    }
    return false;
  }
  return false;
};
/**
 * @param {?} local$$76245
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][computeFrustum] = function(local$$76245) {
  local$$76245[updateMatrixWorld]();
  var local$$76254 = new THREE.Matrix4;
  local$$76254[getInverse](local$$76245[matrixWorld]);
  this[matModelView][multiplyMatrices](local$$76254, this[matLocal]);
  this[matVPW][multiplyMatrices](local$$76245[projectionMatrix], this[matModelView]);
  this[frustum][setFromMatrix](this[matVPW]);
  this[pixelSizeVector] = LSJMath[computePixelSizeVector](this[viewPort], local$$76245[projectionMatrix], this[matModelView]);
};
/**
 * @param {?} local$$76328
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][update] = function(local$$76328) {
  this[lastAccessTime] = (new Date)[getTime]();
  ++this[lastAccessFrame];
  this[computeFrustum](local$$76328);
  /** @type {number} */
  this[curLoadingNode] = 0;
  this[_0x34b6[2143]][update](local$$76328);
  if (this[nodeCount] > this[maxNodeCount]) {
    this[cleanRedundantNodes](this[_0x34b6[2143]]);
  }
};
/**
 * @param {?} local$$76391
 * @return {undefined}
 */
LSJGeoModelLOD[prototype][render] = function(local$$76391) {
  this[meshGroup][visible] = this[visible];
  if (!this[visible]) {
    return;
  }
  this[update](local$$76391[controlCamera]);
};
/**
 * @return {undefined}
 */
LSJFeatureLayer = function() {
  LSJLayer[call](this);
  this[type] = FeatureLayer;
  this[strDataUrl] = ;
  /** @type {!Array} */
  this[_0x34b6[2158]] = [];
};
LSJFeatureLayer[prototype] = Object[create](LSJLayer[prototype]);
/** @type {function(): undefined} */
LSJFeatureLayer[prototype][constructor] = LSJFeatureLayer;
/**
 * @return {undefined}
 */
LSJFeatureLayer[prototype][dispose] = function() {
};
/**
 * @return {?}
 */
LSJFeatureLayer[prototype][getBoundingSphere] = function() {
  return this[boundingSphere];
};
/**
 * @param {?} local$$76507
 * @return {undefined}
 */
LSJFeatureLayer[prototype][addSelectionObject] = function(local$$76507) {
  local$$76507[_0x34b6[2156]](true);
  this[_0x34b6[2158]][push](local$$76507);
};
/**
 * @return {undefined}
 */
LSJFeatureLayer[prototype][releaseSelection] = function() {
  var local$$76540 = this[_0x34b6[2158]][length];
  /** @type {number} */
  var local$$76543 = 0;
  for (; local$$76543 < local$$76540; local$$76543++) {
    var local$$76552 = this[_0x34b6[2158]][local$$76543];
    if (local$$76552 != null) {
      local$$76552[_0x34b6[2156]](false);
    }
  }
};
/**
 * @param {?} local$$76576
 * @return {undefined}
 */
LSJFeatureLayer[prototype][setPath] = function(local$$76576) {
  if (local$$76576 == ) {
    return;
  }
  this[strDataUrl] = local$$76576;
  var local$$76590 = this;
  var local$$76594 = new THREE.XHRLoader;
  local$$76594[load](local$$76576, function(local$$76599) {
    var local$$76605 = JSON[parse](local$$76599);
    var local$$76610 = local$$76605[DataDefine];
    if (local$$76610 !== undefined) {
      if (local$$76610[Range] !== undefined) {
        var local$$76631 = new THREE.Vector3(local$$76610[Range].West, local$$76610[Range].South, local$$76610[Range].MinZ);
        var local$$76647 = new THREE.Vector3(local$$76610[Range].East, local$$76610[Range].North, local$$76610[Range].MaxZ);
        var local$$76651 = new THREE.Vector3;
        local$$76651[set](local$$76631[x] / 2 + local$$76647[x] / 2, local$$76631[y] / 2 + local$$76647[y] / 2, local$$76631[z] / 2 + local$$76647[z] / 2);
        var local$$76693 = new THREE.Vector3;
        local$$76693[subVectors](local$$76647, local$$76631);
        local$$76590[boundingSphere][set](local$$76651, local$$76693[length]() / 2);
      }
      var local$$76720 = local$$76610[_0x34b6[2159]];
      if (local$$76720 !== undefined) {
        var local$$76726 = local$$76720[length];
        /** @type {number} */
        var local$$76729 = 0;
        for (; local$$76729 < local$$76726; local$$76729++) {
          var local$$76735 = local$$76720[local$$76729];
          if (local$$76735 !== undefined) {
            var local$$76739 = new LSJGeoModelLOD;
            local$$76739[setName](local$$76735[_0x34b6[2160]].Name);
            var local$$76763 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2161]][x];
            var local$$76777 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2161]][y];
            var local$$76791 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2161]][z];
            local$$76739[setPosition](local$$76763, local$$76777, local$$76791);
            local$$76739[layer] = local$$76590;
            var local$$76815 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2163]][href];
            local$$76739[_0x34b6[2152]](LSJUtility[getAbsolutePath](LSJUtility[getDir](local$$76576), local$$76815));
            local$$76590[addGeometry](local$$76739);
          }
        }
      }
    }
  });
};
/**
 * @return {undefined}
 */
LSJPointCloudLayer = function() {
  LSJLayer[call](this);
  /** @type {number} */
  this[_0x34b6[2164]] = 1;
  /** @type {number} */
  this[opacity] = 1;
  this[_0x34b6[2165]] = _0x34b6[2166];
  this[_0x34b6[2167]] = Potree[_0x34b6[2169]][_0x34b6[2168]];
  /** @type {null} */
  this[_0x34b6[2170]] = null;
  this[_0x34b6[2171]] = _0x34b6[2172];
  /** @type {number} */
  this[_0x34b6[2173]] = 0;
  /** @type {number} */
  this[_0x34b6[2174]] = 100;
};
LSJPointCloudLayer[prototype] = Object[create](LSJLayer[prototype]);
/** @type {function(): undefined} */
LSJPointCloudLayer[prototype][constructor] = LSJPointCloudLayer;
/**
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][dispose] = function() {
};
/**
 * @param {?} local$$76957
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][_0x34b6[2175]] = function(local$$76957) {
  if (this[_0x34b6[2164]] !== local$$76957) {
    this[_0x34b6[2164]] = local$$76957;
  }
};
/**
 * @return {?}
 */
LSJPointCloudLayer[prototype][_0x34b6[2176]] = function() {
  return this[_0x34b6[2164]];
};
/**
 * @param {?} local$$76995
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][_0x34b6[2177]] = function(local$$76995) {
  if (this[_0x34b6[2165]] !== local$$76995) {
    this[_0x34b6[2165]] = local$$76995;
    if (local$$76995 === _0x34b6[2166]) {
      this[_0x34b6[2167]] = Potree[_0x34b6[2169]][_0x34b6[2168]];
    } else {
      if (local$$76995 === _0x34b6[2178]) {
        this[_0x34b6[2167]] = Potree[_0x34b6[2169]][_0x34b6[2179]];
      } else {
        if (local$$76995 === _0x34b6[2180]) {
          this[_0x34b6[2167]] = Potree[_0x34b6[2169]][_0x34b6[2181]];
        }
      }
    }
  }
};
/**
 * @return {?}
 */
LSJPointCloudLayer[prototype][_0x34b6[2182]] = function() {
  return this[_0x34b6[2165]];
};
/**
 * @param {?} local$$77083
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][_0x34b6[2183]] = function(local$$77083) {
  var local$$77088 = this[_0x34b6[2171]];
  if (local$$77083 == _0x34b6[2184] && !Potree[_0x34b6[2159]][_0x34b6[2186]][_0x34b6[2185]]()) {
    this[_0x34b6[2171]] = _0x34b6[2172];
  } else {
    if (local$$77083 == _0x34b6[2187] && !Potree[_0x34b6[2159]][_0x34b6[2188]][_0x34b6[2185]]()) {
      this[_0x34b6[2171]] = _0x34b6[2172];
    } else {
      this[_0x34b6[2171]] = local$$77083;
    }
  }
};
/**
 * @return {?}
 */
LSJPointCloudLayer[prototype][_0x34b6[2189]] = function() {
  return this[_0x34b6[2171]];
};
/**
 * @param {?} local$$77170
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][_0x34b6[2190]] = function(local$$77170) {
  this[_0x34b6[2191]](local$$77170);
};
/**
 * @return {?}
 */
LSJPointCloudLayer[prototype][_0x34b6[2192]] = function() {
  return this[_0x34b6[2193]](this[_0x34b6[2170]]);
};
/**
 * @param {?} local$$77206
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][_0x34b6[2191]] = function(local$$77206) {
  if (local$$77206 === _0x34b6[2194]) {
    this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2194]];
  } else {
    if (local$$77206 === Color) {
      this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2196]];
    } else {
      if (local$$77206 === _0x34b6[2197]) {
        this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2198]];
      } else {
        if (local$$77206 === _0x34b6[2199]) {
          this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2200]];
        } else {
          if (local$$77206 === _0x34b6[2201]) {
            this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2202]];
          } else {
            if (local$$77206 === _0x34b6[2203]) {
              this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2204]];
            } else {
              if (local$$77206 === _0x34b6[2205]) {
                this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2206]];
              } else {
                if (local$$77206 === _0x34b6[2207]) {
                  this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2208]];
                } else {
                  if (local$$77206 === _0x34b6[2209]) {
                    this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2210]];
                  } else {
                    if (local$$77206 === _0x34b6[2211]) {
                      this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2212]];
                    } else {
                      if (local$$77206 === _0x34b6[2213]) {
                        this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2214]];
                      } else {
                        if (local$$77206 === _0x34b6[2215]) {
                          this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2216]];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};
/**
 * @param {?} local$$77421
 * @return {?}
 */
LSJPointCloudLayer[prototype][_0x34b6[2193]] = function(local$$77421) {
  if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2194]]) {
    return _0x34b6[2194];
  } else {
    if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2196]]) {
      return Color;
    } else {
      if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2198]]) {
        return _0x34b6[2197];
      } else {
        if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2200]]) {
          return _0x34b6[2199];
        } else {
          if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2202]]) {
            return _0x34b6[2201];
          } else {
            if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2204]]) {
              return _0x34b6[2203];
            } else {
              if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2206]]) {
                return _0x34b6[2205];
              } else {
                if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2208]]) {
                  return _0x34b6[2207];
                } else {
                  if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2210]]) {
                    return _0x34b6[2209];
                  } else {
                    if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2212]]) {
                      return _0x34b6[2211];
                    } else {
                      if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2214]]) {
                        return _0x34b6[2213];
                      } else {
                        if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2216]]) {
                          return _0x34b6[2215];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};
/**
 * @param {?} local$$77588
 * @param {?} local$$77589
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][_0x34b6[2217]] = function(local$$77588, local$$77589) {
  if (this[_0x34b6[2173]] !== local$$77588 || this[_0x34b6[2174]] !== local$$77589) {
    this[_0x34b6[2173]] = local$$77588 || this[_0x34b6[2173]];
    this[_0x34b6[2174]] = local$$77589 || this[_0x34b6[2174]];
  }
};
/**
 * @return {?}
 */
LSJPointCloudLayer[prototype][_0x34b6[2218]] = function() {
  return {
    min : this[_0x34b6[2173]],
    max : this[_0x34b6[2174]]
  };
};
/**
 * @return {?}
 */
LSJPointCloudLayer[prototype][getBoundingSphere] = function() {
  return this[boundingSphere];
};
/**
 * @param {?} local$$77664
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][setPath] = function(local$$77664) {
  var local$$77668 = new THREE.Object3D;
  this[meshGroup][add](local$$77668);
  var local$$77678 = this;
  Potree[_0x34b6[2222]][load](local$$77664, function(local$$77686) {
    local$$77678[_0x34b6[2219]] = new Potree.PointCloudOctree(local$$77686);
    local$$77678[_0x34b6[2219]][material][_0x34b6[2220]] = Potree[_0x34b6[2220]][_0x34b6[2221]];
    local$$77668[add](local$$77678[_0x34b6[2219]]);
    local$$77668[updateMatrixWorld](true);
    local$$77678[boundingSphere] = local$$77678[_0x34b6[2219]][boundingSphere][clone]()[applyMatrix4](local$$77678[_0x34b6[2219]][matrixWorld]);
  });
};
/**
 * @param {?} local$$77766
 * @return {undefined}
 */
LSJPointCloudLayer[prototype][render] = function(local$$77766) {
  if (this[_0x34b6[2219]] != undefined) {
    this[_0x34b6[2219]][material][size] = this[_0x34b6[2164]];
    this[_0x34b6[2219]][material][opacity] = this[opacity];
    this[_0x34b6[2219]][material][_0x34b6[2170]] = this[_0x34b6[2170]];
    this[_0x34b6[2219]][material][_0x34b6[2167]] = this[_0x34b6[2167]];
    this[_0x34b6[2219]][material][_0x34b6[2173]] = this[_0x34b6[2173]];
    this[_0x34b6[2219]][material][_0x34b6[2174]] = this[_0x34b6[2174]];
    this[_0x34b6[2219]][update](controlCamera, controlRender);
  }
};
/**
 * @param {?} local$$77873
 * @return {undefined}
 */
LSJGeoLabel = function(local$$77873) {
  LSJGeometry[call](this);
  this[type] = GeoLabel;
  this[position] = new THREE.Vector3(0, 0, 0);
  this[billboard] = undefined;
  /** @type {boolean} */
  this[needUpdate] = true;
  this[screenRect] = new LSJRectangle(0, 0, 0, 0);
  /** @type {number} */
  this[actualAspect] = 1;
  this[material] = undefined;
  this[state] = ;
  /** @type {number} */
  this[_0x34b6[2223]] = 100;
  /** @type {number} */
  this[verticalOrign] = 0;
  /** @type {number} */
  this[horizontalOrigin] = 0;
  this[_0x34b6[2224]] = local$$77873;
  this[update]();
};
LSJGeoLabel[prototype] = Object[create](LSJGeometry[prototype]);
/** @type {function(?): undefined} */
LSJGeoLabel[prototype][constructor] = LSJGeoLabel;
/**
 * @param {?} local$$77994
 * @return {undefined}
 */
LSJGeoLabel[prototype][_0x34b6[2225]] = function(local$$77994) {
  this[verticalOrign] = local$$77994;
};
/**
 * @return {?}
 */
LSJGeoLabel[prototype][_0x34b6[2226]] = function() {
  return this[verticalOrign];
};
/**
 * @return {?}
 */
LSJGeoLabel[prototype][_0x34b6[2227]] = function() {
  return this[horizontalOrigin];
};
/**
 * @param {?} local$$78041
 * @return {undefined}
 */
LSJGeoLabel[prototype][_0x34b6[2228]] = function(local$$78041) {
  this[horizontalOrigin] = local$$78041;
};
/**
 * @param {?} local$$78058
 * @return {undefined}
 */
LSJGeoLabel[prototype][setSize] = function(local$$78058) {
  this[_0x34b6[2223]] = local$$78058;
};
/**
 * @return {?}
 */
LSJGeoLabel[prototype][getSize] = function() {
  return this[_0x34b6[2223]];
};
/**
 * @param {?} local$$78090
 * @return {undefined}
 */
LSJGeoLabel[prototype][_0x34b6[2229]] = function(local$$78090) {
  this[_0x34b6[2224]] = local$$78090;
  /** @type {boolean} */
  this[needUpdate] = true;
};
/**
 * @param {?} local$$78113
 * @return {undefined}
 */
LSJGeoLabel[prototype][setName] = function(local$$78113) {
  this[name] = local$$78113;
};
/**
 * @return {undefined}
 */
LSJGeoLabel[prototype][update] = function() {
  if (this[state] == _0x34b6[2230]) {
    return;
  }
  this[dispose]();
  var local$$78146 = this;
  this[state] = _0x34b6[2230];
  this[_0x34b6[2224]][style][display] = block;
  html2canvas(this[_0x34b6[2224]], {
    allowTaint : true,
    taintTest : false,
    onrendered : function(local$$78175) {
      local$$78146[_0x34b6[2224]][style][display] = none;
      var local$$78194 = local$$78175[toDataURL]();
      /** @type {!Image} */
      var local$$78197 = new Image;
      local$$78197[src] = local$$78194;
      /**
       * @return {undefined}
       */
      local$$78197[onload] = function() {
        var local$$78208 = undefined;
        local$$78208 = document[createElement](canvas);
        local$$78208[width] = local$$78197[width];
        local$$78208[height] = local$$78197[height];
        if (local$$78146[name] == ) {
          /** @type {number} */
          local$$78146[actualAspect] = local$$78208[width] / local$$78208[height];
        }
        var local$$78261 = local$$78208[getContext](2d);
        local$$78261[drawImage](local$$78197, 0, 0, local$$78208[width], local$$78208[height]);
        if (local$$78146[material] == undefined) {
          local$$78146[material] = new THREE.SpriteMaterial({
            depthTest : false,
            map : new THREE.CanvasTexture(local$$78208)
          });
          local$$78146[billboard] = new LSJBillboard(local$$78146[material], local$$78146, {
            verticalOrign : local$$78146[verticalOrign],
            horizontalOrigin : local$$78146[horizontalOrigin]
          });
          local$$78146[billboard][position][copy](local$$78146[position]);
          local$$78146[meshGroup][add](local$$78146[billboard]);
        } else {
          local$$78146[material][map][image] = local$$78208;
          /** @type {boolean} */
          local$$78146[material][map][needsUpdate] = true;
        }
        local$$78146[state] = loaded;
        /** @type {boolean} */
        local$$78146[needUpdate] = false;
      };
    }
  });
};
/**
 * @param {?} local$$78391
 * @param {?} local$$78392
 * @param {?} local$$78393
 * @return {undefined}
 */
LSJGeoLabel[prototype][setPosition] = function(local$$78391, local$$78392, local$$78393) {
  this[position][x] = local$$78391;
  this[position][y] = local$$78392;
  this[position][z] = local$$78393;
  if (this[billboard] != undefined) {
    this[billboard][position][copy](this[position]);
  }
};
/**
 * @return {?}
 */
LSJGeoLabel[prototype][getPosition] = function() {
  return this[postion];
};
/**
 * @return {?}
 */
LSJGeoLabel[prototype][getScreenRect] = function() {
  return this[screenRect];
};
/**
 * @param {string} local$$78479
 * @param {?} local$$78480
 * @return {undefined}
 */
LSJGeoLabel[prototype][render] = function(local$$78479, local$$78480) {
  if (this[needUpdate] && local$$78480) {
    this[update]();
  }
  if (this[billboard] != undefined) {
    this[billboard][camera] = getCamera();
    this[billboard][updateMatrixWorld]();
    local$$78479[controlCamera][updateMatrixWorld]();
    local$$78479[controlCamera][updateProjectionMatrix]();
    var local$$78536 = new THREE.Vector3(0, 0, 0);
    var local$$78543 = new THREE.Vector3(0, 0, 0);
    local$$78543[copy](this[billboard][position]);
    /** @type {number} */
    var local$$78557 = 1;
    /** @type {number} */
    var local$$78560 = 1;
    var local$$78570 = getCamera()[position][distanceTo](local$$78543);
    if (local$$78570 > local$$78479[boundingSphere][radius] && local$$78479[boundingSphere][radius] != 0) {
      var local$$78626 = getCamera()[position][clone]()[add](getCamera()[position][clone]()[sub](local$$78543)[normalize]()[multiplyScalar](local$$78479[boundingSphere][radius]));
      var local$$78639 = local$$78626[clone]()[project](local$$78479[controlCamera]);
      /** @type {number} */
      var local$$78658 = (local$$78639[x] + 1) / 2 * local$$78479[controlRender][domElement][clientWidth];
      var local$$78665 = new THREE.Vector3(0, 1, 0);
      local$$78665[applyQuaternion](local$$78479[controlCamera][quaternion]);
      var local$$78693 = local$$78626[clone]()[add](local$$78665)[project](local$$78479[controlCamera]);
      /** @type {number} */
      var local$$78713 = -(local$$78639[y] - 1) / 2 * local$$78479[controlRender][domElement][clientHeight];
      /** @type {number} */
      var local$$78733 = -(local$$78693[y] - 1) / 2 * local$$78479[controlRender][domElement][clientHeight];
      /** @type {number} */
      var local$$78742 = 1 / Math[abs](local$$78713 - local$$78733);
      local$$78557 = Math[abs](local$$78742);
      local$$78639 = local$$78543[clone]()[project](local$$78479[controlCamera]);
      local$$78665 = new THREE.Vector3(0, 1, 0);
      local$$78665[applyQuaternion](local$$78479[controlCamera][quaternion]);
      local$$78693 = local$$78543[clone]()[add](local$$78665)[project](local$$78479[controlCamera]);
      /** @type {number} */
      local$$78713 = -(local$$78639[y] - 1) / 2 * local$$78479[controlRender][domElement][clientHeight];
      /** @type {number} */
      local$$78733 = -(local$$78693[y] - 1) / 2 * local$$78479[controlRender][domElement][clientHeight];
      /** @type {number} */
      var local$$78845 = 1 / Math[abs](local$$78713 - local$$78733);
      if (local$$78845 > 2 * local$$78742) {
        /** @type {number} */
        local$$78557 = local$$78845 / 2;
      }
      /** @type {number} */
      local$$78560 = local$$78742 / local$$78845;
    } else {
      local$$78639 = local$$78543[clone]()[project](local$$78479[controlCamera]);
      local$$78665 = new THREE.Vector3(0, 1, 0);
      local$$78665[applyQuaternion](local$$78479[controlCamera][quaternion]);
      local$$78693 = local$$78543[clone]()[add](local$$78665)[project](local$$78479[controlCamera]);
      /** @type {number} */
      local$$78713 = -(local$$78639[y] - 1) / 2 * local$$78479[controlRender][domElement][clientHeight];
      /** @type {number} */
      local$$78733 = -(local$$78693[y] - 1) / 2 * local$$78479[controlRender][domElement][clientHeight];
      /** @type {number} */
      local$$78557 = 1 / Math[abs](local$$78713 - local$$78733);
      /** @type {number} */
      local$$78658 = (local$$78639[x] + 1) / 2 * local$$78479[controlRender][domElement][clientWidth];
    }
    if (this[_0x34b6[2224]] != undefined) {
      /** @type {number} */
      this[billboard][scale][x] = this[billboard][scale][y] = this[billboard][scale][z] = this[_0x34b6[2223]] * local$$78557;
      /** @type {number} */
      this[screenRect][left] = local$$78658;
      /** @type {number} */
      this[screenRect][bottom] = local$$78713;
      /** @type {number} */
      this[screenRect][top] = local$$78713 + this[_0x34b6[2223]] * local$$78560;
    }
    /** @type {number} */
    var local$$79064 = this[screenRect][top] - this[screenRect][bottom];
    /** @type {number} */
    var local$$79070 = local$$79064 * this[actualAspect];
    this[screenRect][right] = this[screenRect][left] + local$$79070;
    if (local$$78479 != undefined) {
      local$$78479[billboards][push](this[billboard]);
    }
  }
};
