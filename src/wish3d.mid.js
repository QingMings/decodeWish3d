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
THREE[_0x34b6[0]] = {
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "resolution" : {
      value : new THREE.Vector2(1 / 1024, 1 / 512)
    }
  },
  vertexShader : [_0x34b6[3], _0x34b6[4], _0x34b6[5], _0x34b6[6], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]),
  fragmentShader : [_0x34b6[8], _0x34b6[9], _0x34b6[3], _0x34b6[10], _0x34b6[11], _0x34b6[12], _0x34b6[4], _0x34b6[13], _0x34b6[14], _0x34b6[15], _0x34b6[16], _0x34b6[17], _0x34b6[18], _0x34b6[19], _0x34b6[20], _0x34b6[21], _0x34b6[22], _0x34b6[23], _0x34b6[24], _0x34b6[25], _0x34b6[26], _0x34b6[27], _0x34b6[28], _0x34b6[29], _0x34b6[30], _0x34b6[31], _0x34b6[32], _0x34b6[33], _0x34b6[34], _0x34b6[35], _0x34b6[36], _0x34b6[37], _0x34b6[38], _0x34b6[39], _0x34b6[40], _0x34b6[41], _0x34b6[42], _0x34b6[43], 
  _0x34b6[44], _0x34b6[45], _0x34b6[7], _0x34b6[7]][_0x34b6[2]](_0x34b6[1])
};
/** @type {!Array} */
THREE[_0x34b6[46]] = [{
  defines : {
    "SMAA_THRESHOLD" : _0x34b6[47]
  },
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "resolution" : {
      value : new THREE.Vector2(1 / 1024, 1 / 512)
    }
  },
  vertexShader : [_0x34b6[9], _0x34b6[3], _0x34b6[48], _0x34b6[49], _0x34b6[50], _0x34b6[51], _0x34b6[52], _0x34b6[7], _0x34b6[4], _0x34b6[5], _0x34b6[53], _0x34b6[6], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]),
  fragmentShader : [_0x34b6[8], _0x34b6[3], _0x34b6[48], _0x34b6[54], _0x34b6[55], _0x34b6[56], _0x34b6[57], _0x34b6[58], _0x34b6[59], _0x34b6[60], _0x34b6[61], _0x34b6[62], _0x34b6[63], _0x34b6[64], _0x34b6[65], _0x34b6[66], _0x34b6[67], _0x34b6[68], _0x34b6[69], _0x34b6[70], _0x34b6[71], _0x34b6[72], _0x34b6[73], _0x34b6[74], _0x34b6[75], _0x34b6[69], _0x34b6[76], _0x34b6[77], _0x34b6[72], _0x34b6[78], _0x34b6[79], _0x34b6[80], _0x34b6[7], _0x34b6[4], _0x34b6[81], _0x34b6[7]][_0x34b6[2]](_0x34b6[1])
}, {
  defines : {
    "SMAA_MAX_SEARCH_STEPS" : _0x34b6[82],
    "SMAA_AREATEX_MAX_DISTANCE" : _0x34b6[83],
    "SMAA_AREATEX_PIXEL_SIZE" : _0x34b6[84],
    "SMAA_AREATEX_SUBTEX_SIZE" : _0x34b6[85]
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
  vertexShader : [_0x34b6[9], _0x34b6[3], _0x34b6[48], _0x34b6[86], _0x34b6[87], _0x34b6[88], _0x34b6[89], _0x34b6[90], _0x34b6[91], _0x34b6[7], _0x34b6[4], _0x34b6[5], _0x34b6[92], _0x34b6[6], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]),
  fragmentShader : [_0x34b6[93], _0x34b6[8], _0x34b6[94], _0x34b6[95], _0x34b6[9], _0x34b6[3], _0x34b6[96], _0x34b6[86], _0x34b6[97], _0x34b6[98], _0x34b6[7], _0x34b6[99], _0x34b6[100], _0x34b6[101], _0x34b6[7], _0x34b6[102], _0x34b6[103], _0x34b6[104], _0x34b6[105], _0x34b6[106], _0x34b6[107], _0x34b6[7], _0x34b6[108], _0x34b6[109], _0x34b6[110], _0x34b6[111], _0x34b6[112], _0x34b6[7], _0x34b6[113], _0x34b6[103], _0x34b6[104], _0x34b6[105], _0x34b6[114], _0x34b6[115], _0x34b6[7], _0x34b6[116], _0x34b6[117], 
  _0x34b6[118], _0x34b6[119], _0x34b6[112], _0x34b6[7], _0x34b6[120], _0x34b6[121], _0x34b6[104], _0x34b6[105], _0x34b6[122], _0x34b6[123], _0x34b6[7], _0x34b6[124], _0x34b6[125], _0x34b6[126], _0x34b6[127], _0x34b6[128], _0x34b6[7], _0x34b6[129], _0x34b6[121], _0x34b6[104], _0x34b6[105], _0x34b6[130], _0x34b6[131], _0x34b6[7], _0x34b6[132], _0x34b6[133], _0x34b6[134], _0x34b6[135], _0x34b6[128], _0x34b6[7], _0x34b6[136], _0x34b6[137], _0x34b6[138], _0x34b6[139], _0x34b6[140], _0x34b6[7], _0x34b6[141], 
  _0x34b6[142], _0x34b6[143], _0x34b6[144], _0x34b6[145], _0x34b6[146], _0x34b6[147], _0x34b6[148], _0x34b6[149], _0x34b6[150], _0x34b6[151], _0x34b6[152], _0x34b6[153], _0x34b6[154], _0x34b6[155], _0x34b6[156], _0x34b6[157], _0x34b6[7], _0x34b6[158], _0x34b6[145], _0x34b6[146], _0x34b6[159], _0x34b6[160], _0x34b6[161], _0x34b6[162], _0x34b6[163], _0x34b6[164], _0x34b6[165], _0x34b6[154], _0x34b6[155], _0x34b6[166], _0x34b6[167], _0x34b6[7], _0x34b6[168], _0x34b6[7], _0x34b6[4], _0x34b6[169], _0x34b6[7]][_0x34b6[2]](_0x34b6[1])
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
  vertexShader : [_0x34b6[9], _0x34b6[3], _0x34b6[170], _0x34b6[171], _0x34b6[172], _0x34b6[173], _0x34b6[7], _0x34b6[4], _0x34b6[5], _0x34b6[174], _0x34b6[6], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]),
  fragmentShader : [_0x34b6[8], _0x34b6[175], _0x34b6[9], _0x34b6[3], _0x34b6[170], _0x34b6[176], _0x34b6[177], _0x34b6[178], _0x34b6[179], _0x34b6[180], _0x34b6[181], _0x34b6[182], _0x34b6[44], _0x34b6[183], _0x34b6[184], _0x34b6[185], _0x34b6[186], _0x34b6[187], _0x34b6[44], _0x34b6[188], _0x34b6[7], _0x34b6[189], _0x34b6[190], _0x34b6[191], _0x34b6[192], _0x34b6[193], _0x34b6[194], _0x34b6[195], _0x34b6[196], _0x34b6[197], _0x34b6[7], _0x34b6[7], _0x34b6[4], _0x34b6[198], _0x34b6[7]][_0x34b6[2]](_0x34b6[1])
}];
THREE[_0x34b6[199]] = {
  uniforms : {
    "tDiffuse" : {
      value : null
    },
    "opacity" : {
      value : 1
    }
  },
  vertexShader : [_0x34b6[3], _0x34b6[4], _0x34b6[5], _0x34b6[6], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]),
  fragmentShader : [_0x34b6[200], _0x34b6[8], _0x34b6[3], _0x34b6[4], _0x34b6[201], _0x34b6[202], _0x34b6[7]][_0x34b6[2]](_0x34b6[1])
};
/**
 * @param {?} local$$2979
 * @param {!Array} local$$2980
 * @return {undefined}
 */
THREE[_0x34b6[203]] = function(local$$2979, local$$2980) {
  this[_0x34b6[204]] = local$$2979;
  if (local$$2980 === undefined) {
    var local$$3002 = {
      minFilter : THREE[_0x34b6[205]],
      magFilter : THREE[_0x34b6[205]],
      format : THREE[_0x34b6[206]],
      stencilBuffer : false
    };
    var local$$3008 = local$$2979[_0x34b6[207]]();
    local$$2980 = new THREE.WebGLRenderTarget(local$$3008[_0x34b6[208]], local$$3008[_0x34b6[209]], local$$3002);
  }
  /** @type {!Array} */
  this[_0x34b6[210]] = local$$2980;
  this[_0x34b6[211]] = local$$2980[_0x34b6[212]]();
  this[_0x34b6[213]] = this[_0x34b6[210]];
  this[_0x34b6[214]] = this[_0x34b6[211]];
  /** @type {!Array} */
  this[_0x34b6[215]] = [];
  if (THREE[_0x34b6[199]] === undefined) {
    console[_0x34b6[217]](_0x34b6[216]);
  }
  this[_0x34b6[218]] = new THREE.ShaderPass(THREE.CopyShader);
};
Object[_0x34b6[233]](THREE[_0x34b6[203]][_0x34b6[219]], {
  swapBuffers : function() {
    var local$$3101 = this[_0x34b6[214]];
    this[_0x34b6[214]] = this[_0x34b6[213]];
    this[_0x34b6[213]] = local$$3101;
  },
  addPass : function(local$$3118) {
    this[_0x34b6[215]][_0x34b6[220]](local$$3118);
    var local$$3135 = this[_0x34b6[204]][_0x34b6[207]]();
    local$$3118[_0x34b6[221]](local$$3135[_0x34b6[208]], local$$3135[_0x34b6[209]]);
  },
  insertPass : function(local$$3150, local$$3151) {
    this[_0x34b6[215]][_0x34b6[222]](local$$3151, 0, local$$3150);
  },
  render : function(local$$3165) {
    /** @type {boolean} */
    var local$$3168 = false;
    var local$$3170;
    var local$$3172;
    var local$$3180 = this[_0x34b6[215]][_0x34b6[223]];
    /** @type {number} */
    local$$3172 = 0;
    for (; local$$3172 < local$$3180; local$$3172++) {
      local$$3170 = this[_0x34b6[215]][local$$3172];
      if (local$$3170[_0x34b6[224]] === false) {
        continue;
      }
      local$$3170[_0x34b6[225]](this[_0x34b6[204]], this[_0x34b6[213]], this[_0x34b6[214]], local$$3165, local$$3168);
      if (local$$3170[_0x34b6[226]]) {
        if (local$$3168) {
          var local$$3226 = this[_0x34b6[204]][_0x34b6[227]];
          local$$3226[_0x34b6[228]](local$$3226.NOTEQUAL, 1, 4294967295);
          this[_0x34b6[218]][_0x34b6[225]](this[_0x34b6[204]], this[_0x34b6[213]], this[_0x34b6[214]], local$$3165);
          local$$3226[_0x34b6[228]](local$$3226.EQUAL, 1, 4294967295);
        }
        this[_0x34b6[229]]();
      }
      if (THREE[_0x34b6[230]] !== undefined) {
        if (local$$3170 instanceof THREE[_0x34b6[230]]) {
          /** @type {boolean} */
          local$$3168 = true;
        } else {
          if (local$$3170 instanceof THREE[_0x34b6[231]]) {
            /** @type {boolean} */
            local$$3168 = false;
          }
        }
      }
    }
  },
  reset : function(local$$3303) {
    if (local$$3303 === undefined) {
      var local$$3313 = this[_0x34b6[204]][_0x34b6[207]]();
      local$$3303 = this[_0x34b6[210]][_0x34b6[212]]();
      local$$3303[_0x34b6[221]](local$$3313[_0x34b6[208]], local$$3313[_0x34b6[209]]);
    }
    this[_0x34b6[210]][_0x34b6[232]]();
    this[_0x34b6[211]][_0x34b6[232]]();
    /** @type {!Array} */
    this[_0x34b6[210]] = local$$3303;
    this[_0x34b6[211]] = local$$3303[_0x34b6[212]]();
    this[_0x34b6[213]] = this[_0x34b6[210]];
    this[_0x34b6[214]] = this[_0x34b6[211]];
  },
  setSize : function(local$$3386, local$$3387) {
    this[_0x34b6[210]][_0x34b6[221]](local$$3386, local$$3387);
    this[_0x34b6[211]][_0x34b6[221]](local$$3386, local$$3387);
    /** @type {number} */
    var local$$3406 = 0;
    for (; local$$3406 < this[_0x34b6[215]][_0x34b6[223]]; local$$3406++) {
      this[_0x34b6[215]][local$$3406][_0x34b6[221]](local$$3386, local$$3387);
    }
  }
});
/**
 * @return {undefined}
 */
THREE[_0x34b6[234]] = function() {
  /** @type {boolean} */
  this[_0x34b6[224]] = true;
  /** @type {boolean} */
  this[_0x34b6[226]] = true;
  /** @type {boolean} */
  this[_0x34b6[235]] = false;
  /** @type {boolean} */
  this[_0x34b6[236]] = false;
};
Object[_0x34b6[233]](THREE[_0x34b6[234]][_0x34b6[219]], {
  setSize : function(local$$3474, local$$3475) {
  },
  render : function(local$$3479, local$$3480, local$$3481, local$$3482, local$$3483) {
    console[_0x34b6[217]](_0x34b6[237]);
  }
});
/**
 * @param {?} local$$3502
 * @param {?} local$$3503
 * @return {undefined}
 */
THREE[_0x34b6[230]] = function(local$$3502, local$$3503) {
  THREE[_0x34b6[234]][_0x34b6[238]](this);
  this[_0x34b6[239]] = local$$3502;
  this[_0x34b6[240]] = local$$3503;
  /** @type {boolean} */
  this[_0x34b6[235]] = true;
  /** @type {boolean} */
  this[_0x34b6[226]] = false;
  /** @type {boolean} */
  this[_0x34b6[241]] = false;
};
THREE[_0x34b6[230]][_0x34b6[219]] = Object[_0x34b6[233]](Object[_0x34b6[242]](THREE[_0x34b6[234]][_0x34b6[219]]), {
  constructor : THREE[_0x34b6[230]],
  render : function(local$$3567, local$$3568, local$$3569, local$$3570, local$$3571) {
    var local$$3576 = local$$3567[_0x34b6[227]];
    var local$$3581 = local$$3567[_0x34b6[243]];
    local$$3581[_0x34b6[246]][_0x34b6[245]][_0x34b6[244]](false);
    local$$3581[_0x34b6[246]][_0x34b6[247]][_0x34b6[244]](false);
    local$$3581[_0x34b6[246]][_0x34b6[245]][_0x34b6[248]](true);
    local$$3581[_0x34b6[246]][_0x34b6[247]][_0x34b6[248]](true);
    var local$$3631;
    var local$$3633;
    if (this[_0x34b6[241]]) {
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
    local$$3581[_0x34b6[246]][_0x34b6[250]][_0x34b6[249]](true);
    local$$3581[_0x34b6[246]][_0x34b6[250]][_0x34b6[251]](local$$3576.REPLACE, local$$3576.REPLACE, local$$3576.REPLACE);
    local$$3581[_0x34b6[246]][_0x34b6[250]][_0x34b6[252]](local$$3576.ALWAYS, local$$3631, 4294967295);
    local$$3581[_0x34b6[246]][_0x34b6[250]][_0x34b6[253]](local$$3633);
    local$$3567[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], local$$3569, this[_0x34b6[235]]);
    local$$3567[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], local$$3568, this[_0x34b6[235]]);
    local$$3581[_0x34b6[246]][_0x34b6[245]][_0x34b6[248]](false);
    local$$3581[_0x34b6[246]][_0x34b6[247]][_0x34b6[248]](false);
    local$$3581[_0x34b6[246]][_0x34b6[250]][_0x34b6[252]](local$$3576.EQUAL, 1, 4294967295);
    local$$3581[_0x34b6[246]][_0x34b6[250]][_0x34b6[251]](local$$3576.KEEP, local$$3576.KEEP, local$$3576.KEEP);
  }
});
/**
 * @return {undefined}
 */
THREE[_0x34b6[231]] = function() {
  THREE[_0x34b6[234]][_0x34b6[238]](this);
  /** @type {boolean} */
  this[_0x34b6[226]] = false;
};
THREE[_0x34b6[231]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[234]][_0x34b6[219]]);
Object[_0x34b6[233]](THREE[_0x34b6[231]][_0x34b6[219]], {
  render : function(local$$3842, local$$3843, local$$3844, local$$3845, local$$3846) {
    local$$3842[_0x34b6[243]][_0x34b6[246]][_0x34b6[250]][_0x34b6[249]](false);
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
THREE[_0x34b6[254]] = function(local$$3873, local$$3874, local$$3875, local$$3876, local$$3877) {
  THREE[_0x34b6[234]][_0x34b6[238]](this);
  this[_0x34b6[239]] = local$$3873;
  this[_0x34b6[240]] = local$$3874;
  this[_0x34b6[255]] = local$$3875;
  this[_0x34b6[256]] = local$$3876;
  this[_0x34b6[257]] = local$$3877 !== undefined ? local$$3877 : 0;
  /** @type {boolean} */
  this[_0x34b6[235]] = true;
  /** @type {boolean} */
  this[_0x34b6[226]] = false;
};
THREE[_0x34b6[254]][_0x34b6[219]] = Object[_0x34b6[233]](Object[_0x34b6[242]](THREE[_0x34b6[234]][_0x34b6[219]]), {
  constructor : THREE[_0x34b6[254]],
  render : function(local$$3953, local$$3954, local$$3955, local$$3956, local$$3957) {
    var local$$3962 = local$$3953[_0x34b6[258]];
    /** @type {boolean} */
    local$$3953[_0x34b6[258]] = false;
    this[_0x34b6[239]][_0x34b6[255]] = this[_0x34b6[255]];
    var local$$3981;
    var local$$3983;
    if (this[_0x34b6[256]]) {
      local$$3981 = local$$3953[_0x34b6[260]]()[_0x34b6[259]]();
      local$$3983 = local$$3953[_0x34b6[261]]();
      local$$3953[_0x34b6[262]](this[_0x34b6[256]], this[_0x34b6[257]]);
    }
    local$$3953[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[236]] ? null : local$$3955, this[_0x34b6[235]]);
    if (this[_0x34b6[256]]) {
      local$$3953[_0x34b6[262]](local$$3981, local$$3983);
    }
    /** @type {null} */
    this[_0x34b6[239]][_0x34b6[255]] = null;
    local$$3953[_0x34b6[258]] = local$$3962;
  }
});
/**
 * @param {?} local$$4073
 * @param {string} local$$4074
 * @return {undefined}
 */
THREE[_0x34b6[263]] = function(local$$4073, local$$4074) {
  THREE[_0x34b6[234]][_0x34b6[238]](this);
  this[_0x34b6[264]] = local$$4074 !== undefined ? local$$4074 : _0x34b6[265];
  if (local$$4073 instanceof THREE[_0x34b6[266]]) {
    this[_0x34b6[267]] = local$$4073[_0x34b6[267]];
    this[_0x34b6[268]] = local$$4073;
  } else {
    if (local$$4073) {
      this[_0x34b6[267]] = THREE[_0x34b6[269]][_0x34b6[212]](local$$4073[_0x34b6[267]]);
      this[_0x34b6[268]] = new THREE.ShaderMaterial({
        defines : local$$4073[_0x34b6[270]] || {},
        uniforms : this[_0x34b6[267]],
        vertexShader : local$$4073[_0x34b6[271]],
        fragmentShader : local$$4073[_0x34b6[272]]
      });
    }
  }
  this[_0x34b6[240]] = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  this[_0x34b6[239]] = new THREE.Scene;
  this[_0x34b6[273]] = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), null);
  this[_0x34b6[239]][_0x34b6[274]](this[_0x34b6[273]]);
};
THREE[_0x34b6[263]][_0x34b6[219]] = Object[_0x34b6[233]](Object[_0x34b6[242]](THREE[_0x34b6[234]][_0x34b6[219]]), {
  constructor : THREE[_0x34b6[263]],
  render : function(local$$4223, local$$4224, local$$4225, local$$4226, local$$4227) {
    if (this[_0x34b6[267]][this[_0x34b6[264]]]) {
      this[_0x34b6[267]][this[_0x34b6[264]]][_0x34b6[275]] = local$$4225[_0x34b6[276]];
    }
    this[_0x34b6[273]][_0x34b6[268]] = this[_0x34b6[268]];
    if (this[_0x34b6[236]]) {
      local$$4223[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]]);
    } else {
      local$$4223[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], local$$4224, this[_0x34b6[235]]);
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
THREE[_0x34b6[277]] = function(local$$4307, local$$4308, local$$4309, local$$4310) {
  this[_0x34b6[278]] = local$$4308;
  this[_0x34b6[279]] = local$$4309;
  this[_0x34b6[280]] = local$$4310 !== undefined ? local$$4310 : [];
  this[_0x34b6[281]] = new THREE.Color(1, 1, 1);
  this[_0x34b6[282]] = new THREE.Color(.1, .04, .02);
  /** @type {number} */
  this[_0x34b6[283]] = .4;
  /** @type {boolean} */
  this[_0x34b6[284]] = false;
  /** @type {number} */
  this[_0x34b6[285]] = 1.4;
  /** @type {number} */
  this[_0x34b6[286]] = 8;
  /** @type {number} */
  this[_0x34b6[287]] = 1;
  /** @type {number} */
  this[_0x34b6[288]] = 2;
  THREE[_0x34b6[234]][_0x34b6[238]](this);
  this[_0x34b6[289]] = local$$4307 !== undefined ? new THREE.Vector2(local$$4307[_0x34b6[290]], local$$4307[_0x34b6[291]]) : new THREE.Vector2(256, 256);
  var local$$4423 = {
    minFilter : THREE[_0x34b6[205]],
    magFilter : THREE[_0x34b6[205]],
    format : THREE[_0x34b6[206]]
  };
  var local$$4440 = Math[_0x34b6[292]](this[_0x34b6[289]][_0x34b6[290]] / this[_0x34b6[287]]);
  var local$$4456 = Math[_0x34b6[292]](this[_0x34b6[289]][_0x34b6[291]] / this[_0x34b6[287]]);
  this[_0x34b6[293]] = new THREE.MeshBasicMaterial({
    color : 16777215
  });
  this[_0x34b6[293]][_0x34b6[294]] = THREE[_0x34b6[295]];
  this[_0x34b6[296]] = new THREE.WebGLRenderTarget(this[_0x34b6[289]][_0x34b6[290]], this[_0x34b6[289]][_0x34b6[291]], local$$4423);
  /** @type {boolean} */
  this[_0x34b6[296]][_0x34b6[276]][_0x34b6[297]] = false;
  this[_0x34b6[298]] = new THREE.MeshDepthMaterial;
  this[_0x34b6[298]][_0x34b6[294]] = THREE[_0x34b6[295]];
  this[_0x34b6[298]][_0x34b6[299]] = THREE[_0x34b6[300]];
  this[_0x34b6[298]][_0x34b6[301]] = THREE[_0x34b6[302]];
  this[_0x34b6[303]] = this[_0x34b6[304]]();
  this[_0x34b6[303]][_0x34b6[294]] = THREE[_0x34b6[295]];
  this[_0x34b6[305]] = new THREE.WebGLRenderTarget(this[_0x34b6[289]][_0x34b6[290]], this[_0x34b6[289]][_0x34b6[291]], local$$4423);
  /** @type {boolean} */
  this[_0x34b6[305]][_0x34b6[276]][_0x34b6[297]] = false;
  this[_0x34b6[306]] = new THREE.WebGLRenderTarget(local$$4440, local$$4456, local$$4423);
  /** @type {boolean} */
  this[_0x34b6[306]][_0x34b6[276]][_0x34b6[297]] = false;
  this[_0x34b6[307]] = new THREE.WebGLRenderTarget(local$$4440, local$$4456, local$$4423);
  /** @type {boolean} */
  this[_0x34b6[307]][_0x34b6[276]][_0x34b6[297]] = false;
  this[_0x34b6[308]] = new THREE.WebGLRenderTarget(Math[_0x34b6[292]](local$$4440 / 2), Math[_0x34b6[292]](local$$4456 / 2), local$$4423);
  /** @type {boolean} */
  this[_0x34b6[308]][_0x34b6[276]][_0x34b6[297]] = false;
  this[_0x34b6[309]] = this[_0x34b6[310]]();
  this[_0x34b6[311]] = new THREE.WebGLRenderTarget(local$$4440, local$$4456, local$$4423);
  /** @type {boolean} */
  this[_0x34b6[311]][_0x34b6[276]][_0x34b6[297]] = false;
  this[_0x34b6[312]] = new THREE.WebGLRenderTarget(Math[_0x34b6[292]](local$$4440 / 2), Math[_0x34b6[292]](local$$4456 / 2), local$$4423);
  /** @type {boolean} */
  this[_0x34b6[312]][_0x34b6[276]][_0x34b6[297]] = false;
  /** @type {number} */
  var local$$4730 = 4;
  /** @type {number} */
  var local$$4733 = 4;
  this[_0x34b6[313]] = this[_0x34b6[314]](local$$4730);
  this[_0x34b6[313]][_0x34b6[267]][_0x34b6[315]][_0x34b6[275]] = new THREE.Vector2(local$$4440, local$$4456);
  /** @type {number} */
  this[_0x34b6[313]][_0x34b6[267]][_0x34b6[316]][_0x34b6[275]] = 1;
  this[_0x34b6[317]] = this[_0x34b6[314]](local$$4733);
  this[_0x34b6[317]][_0x34b6[267]][_0x34b6[315]][_0x34b6[275]] = new THREE.Vector2(Math[_0x34b6[292]](local$$4440 / 2), Math[_0x34b6[292]](local$$4456 / 2));
  /** @type {number} */
  this[_0x34b6[317]][_0x34b6[267]][_0x34b6[316]][_0x34b6[275]] = local$$4733;
  this[_0x34b6[318]] = this[_0x34b6[319]]();
  if (THREE[_0x34b6[199]] === undefined) {
    console[_0x34b6[217]](_0x34b6[320]);
  }
  var local$$4852 = THREE[_0x34b6[199]];
  this[_0x34b6[321]] = THREE[_0x34b6[269]][_0x34b6[212]](local$$4852[_0x34b6[267]]);
  /** @type {number} */
  this[_0x34b6[321]][_0x34b6[322]][_0x34b6[275]] = 1;
  this[_0x34b6[323]] = new THREE.ShaderMaterial({
    uniforms : this[_0x34b6[321]],
    vertexShader : local$$4852[_0x34b6[271]],
    fragmentShader : local$$4852[_0x34b6[272]],
    blending : THREE[_0x34b6[302]],
    depthTest : false,
    depthWrite : false,
    transparent : true
  });
  /** @type {boolean} */
  this[_0x34b6[224]] = true;
  /** @type {boolean} */
  this[_0x34b6[226]] = false;
  this[_0x34b6[324]] = new THREE.Color;
  /** @type {number} */
  this[_0x34b6[325]] = 1;
  this[_0x34b6[240]] = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  this[_0x34b6[239]] = new THREE.Scene;
  this[_0x34b6[273]] = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2), null);
  this[_0x34b6[239]][_0x34b6[274]](this[_0x34b6[273]]);
  this[_0x34b6[326]] = new THREE.Color;
  this[_0x34b6[327]] = new THREE.Color;
  this[_0x34b6[328]] = new THREE.Matrix4;
};
THREE[_0x34b6[277]][_0x34b6[219]] = Object[_0x34b6[233]](Object[_0x34b6[242]](THREE[_0x34b6[234]][_0x34b6[219]]), {
  constructor : THREE[_0x34b6[277]],
  dispose : function() {
    this[_0x34b6[296]][_0x34b6[232]]();
    this[_0x34b6[305]][_0x34b6[232]]();
    this[_0x34b6[306]][_0x34b6[232]]();
    this[_0x34b6[307]][_0x34b6[232]]();
    this[_0x34b6[308]][_0x34b6[232]]();
    this[_0x34b6[311]][_0x34b6[232]]();
    this[_0x34b6[312]][_0x34b6[232]]();
  },
  setSize : function(local$$5079, local$$5080) {
    this[_0x34b6[296]][_0x34b6[221]](local$$5079, local$$5080);
    var local$$5098 = Math[_0x34b6[292]](local$$5079 / this[_0x34b6[287]]);
    var local$$5108 = Math[_0x34b6[292]](local$$5080 / this[_0x34b6[287]]);
    this[_0x34b6[306]][_0x34b6[221]](local$$5098, local$$5108);
    this[_0x34b6[307]][_0x34b6[221]](local$$5098, local$$5108);
    this[_0x34b6[311]][_0x34b6[221]](local$$5098, local$$5108);
    this[_0x34b6[313]][_0x34b6[267]][_0x34b6[315]][_0x34b6[275]] = new THREE.Vector2(local$$5098, local$$5108);
    local$$5098 = Math[_0x34b6[292]](local$$5098 / 2);
    local$$5108 = Math[_0x34b6[292]](local$$5108 / 2);
    this[_0x34b6[308]][_0x34b6[221]](local$$5098, local$$5108);
    this[_0x34b6[312]][_0x34b6[221]](local$$5098, local$$5108);
    this[_0x34b6[317]][_0x34b6[267]][_0x34b6[315]][_0x34b6[275]] = new THREE.Vector2(local$$5098, local$$5108);
  },
  changeVisibilityOfSelectedObjects : function(local$$5200) {
    /**
     * @param {?} local$$5202
     * @return {undefined}
     */
    var local$$5217 = function(local$$5202) {
      if (local$$5202 instanceof THREE[_0x34b6[329]]) {
        local$$5202[_0x34b6[330]] = local$$5200;
      }
    };
    /** @type {number} */
    var local$$5220 = 0;
    for (; local$$5220 < this[_0x34b6[280]][_0x34b6[223]]; local$$5220++) {
      var local$$5235 = this[_0x34b6[280]][local$$5220];
      local$$5235[_0x34b6[331]](local$$5217);
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
      if (local$$5251 instanceof THREE[_0x34b6[329]]) {
        local$$5249[_0x34b6[220]](local$$5251);
      }
    };
    /** @type {number} */
    var local$$5269 = 0;
    for (; local$$5269 < this[_0x34b6[280]][_0x34b6[223]]; local$$5269++) {
      var local$$5284 = this[_0x34b6[280]][local$$5269];
      local$$5284[_0x34b6[331]](local$$5266);
    }
    /**
     * @param {?} local$$5294
     * @return {undefined}
     */
    var local$$5361 = function(local$$5294) {
      if (local$$5294 instanceof THREE[_0x34b6[329]]) {
        /** @type {boolean} */
        var local$$5301 = false;
        /** @type {number} */
        var local$$5304 = 0;
        for (; local$$5304 < local$$5249[_0x34b6[223]]; local$$5304++) {
          var local$$5316 = local$$5249[local$$5304][_0x34b6[332]];
          if (local$$5316 === local$$5294[_0x34b6[332]]) {
            /** @type {boolean} */
            local$$5301 = true;
            break;
          }
        }
        if (!local$$5301) {
          var local$$5335 = local$$5294[_0x34b6[330]];
          if (!local$$5246 || local$$5294[_0x34b6[333]]) {
            local$$5294[_0x34b6[330]] = local$$5246;
          }
          local$$5294[_0x34b6[333]] = local$$5335;
        }
      }
    };
    this[_0x34b6[278]][_0x34b6[331]](local$$5361);
  },
  updateTextureMatrix : function() {
    this[_0x34b6[328]][_0x34b6[334]](.5, 0, 0, .5, 0, .5, 0, .5, 0, 0, .5, .5, 0, 0, 0, 1);
    this[_0x34b6[328]][_0x34b6[336]](this[_0x34b6[279]][_0x34b6[335]]);
    this[_0x34b6[328]][_0x34b6[336]](this[_0x34b6[279]][_0x34b6[337]]);
  },
  render : function(local$$5428, local$$5429, local$$5430, local$$5431, local$$5432) {
    if (this[_0x34b6[280]][_0x34b6[223]] === 0) {
      return;
    }
    this[_0x34b6[324]][_0x34b6[338]](local$$5428[_0x34b6[260]]());
    this[_0x34b6[325]] = local$$5428[_0x34b6[261]]();
    var local$$5470 = local$$5428[_0x34b6[258]];
    /** @type {boolean} */
    local$$5428[_0x34b6[258]] = false;
    if (local$$5432) {
      local$$5428[_0x34b6[227]][_0x34b6[339]](local$$5428[_0x34b6[227]].STENCIL_TEST);
    }
    local$$5428[_0x34b6[262]](16777215, 1);
    this[_0x34b6[340]](false);
    this[_0x34b6[278]][_0x34b6[255]] = this[_0x34b6[298]];
    local$$5428[_0x34b6[225]](this[_0x34b6[278]], this[_0x34b6[279]], this[_0x34b6[305]], true);
    this[_0x34b6[340]](true);
    this[_0x34b6[341]]();
    this[_0x34b6[342]](false);
    var local$$5556 = this[_0x34b6[278]][_0x34b6[343]];
    /** @type {null} */
    this[_0x34b6[278]][_0x34b6[343]] = null;
    this[_0x34b6[278]][_0x34b6[255]] = this[_0x34b6[303]];
    this[_0x34b6[303]][_0x34b6[267]][_0x34b6[344]][_0x34b6[275]] = new THREE.Vector2(this[_0x34b6[279]][_0x34b6[345]], this[_0x34b6[279]][_0x34b6[346]]);
    this[_0x34b6[303]][_0x34b6[267]][_0x34b6[347]][_0x34b6[275]] = this[_0x34b6[305]][_0x34b6[276]];
    this[_0x34b6[303]][_0x34b6[267]][_0x34b6[328]][_0x34b6[275]] = this[_0x34b6[328]];
    local$$5428[_0x34b6[225]](this[_0x34b6[278]], this[_0x34b6[279]], this[_0x34b6[296]], true);
    /** @type {null} */
    this[_0x34b6[278]][_0x34b6[255]] = null;
    this[_0x34b6[342]](true);
    this[_0x34b6[278]][_0x34b6[343]] = local$$5556;
    this[_0x34b6[273]][_0x34b6[268]] = this[_0x34b6[323]];
    this[_0x34b6[321]][_0x34b6[265]][_0x34b6[275]] = this[_0x34b6[296]][_0x34b6[276]];
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[306]], true);
    this[_0x34b6[326]][_0x34b6[338]](this[_0x34b6[281]]);
    this[_0x34b6[327]][_0x34b6[338]](this[_0x34b6[282]]);
    if (this[_0x34b6[288]] > 0) {
      /** @type {number} */
      var local$$5778 = (1 + .25) / 2 + Math[_0x34b6[349]](performance[_0x34b6[348]]() * .01 / this[_0x34b6[288]]) * (1 - .25) / 2;
      this[_0x34b6[326]][_0x34b6[350]](local$$5778);
      this[_0x34b6[327]][_0x34b6[350]](local$$5778);
    }
    this[_0x34b6[273]][_0x34b6[268]] = this[_0x34b6[309]];
    this[_0x34b6[309]][_0x34b6[267]][_0x34b6[351]][_0x34b6[275]] = this[_0x34b6[306]][_0x34b6[276]];
    this[_0x34b6[309]][_0x34b6[267]][_0x34b6[315]][_0x34b6[275]] = new THREE.Vector2(this[_0x34b6[306]][_0x34b6[208]], this[_0x34b6[306]][_0x34b6[209]]);
    this[_0x34b6[309]][_0x34b6[267]][_0x34b6[281]][_0x34b6[275]] = this[_0x34b6[326]];
    this[_0x34b6[309]][_0x34b6[267]][_0x34b6[282]][_0x34b6[275]] = this[_0x34b6[327]];
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[311]], true);
    this[_0x34b6[273]][_0x34b6[268]] = this[_0x34b6[313]];
    this[_0x34b6[313]][_0x34b6[267]][_0x34b6[352]][_0x34b6[275]] = this[_0x34b6[311]][_0x34b6[276]];
    this[_0x34b6[313]][_0x34b6[267]][_0x34b6[353]][_0x34b6[275]] = THREE[_0x34b6[277]][_0x34b6[354]];
    this[_0x34b6[313]][_0x34b6[267]][_0x34b6[316]][_0x34b6[275]] = this[_0x34b6[285]];
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[307]], true);
    this[_0x34b6[313]][_0x34b6[267]][_0x34b6[352]][_0x34b6[275]] = this[_0x34b6[307]][_0x34b6[276]];
    this[_0x34b6[313]][_0x34b6[267]][_0x34b6[353]][_0x34b6[275]] = THREE[_0x34b6[277]][_0x34b6[355]];
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[311]], true);
    this[_0x34b6[273]][_0x34b6[268]] = this[_0x34b6[317]];
    this[_0x34b6[317]][_0x34b6[267]][_0x34b6[352]][_0x34b6[275]] = this[_0x34b6[311]][_0x34b6[276]];
    this[_0x34b6[317]][_0x34b6[267]][_0x34b6[353]][_0x34b6[275]] = THREE[_0x34b6[277]][_0x34b6[354]];
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[308]], true);
    this[_0x34b6[317]][_0x34b6[267]][_0x34b6[352]][_0x34b6[275]] = this[_0x34b6[308]][_0x34b6[276]];
    this[_0x34b6[317]][_0x34b6[267]][_0x34b6[353]][_0x34b6[275]] = THREE[_0x34b6[277]][_0x34b6[355]];
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], this[_0x34b6[312]], true);
    this[_0x34b6[273]][_0x34b6[268]] = this[_0x34b6[318]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[351]][_0x34b6[275]] = this[_0x34b6[296]][_0x34b6[276]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[356]][_0x34b6[275]] = this[_0x34b6[311]][_0x34b6[276]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[357]][_0x34b6[275]] = this[_0x34b6[312]][_0x34b6[276]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[358]][_0x34b6[275]] = this[_0x34b6[358]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[286]][_0x34b6[275]] = this[_0x34b6[286]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[283]][_0x34b6[275]] = this[_0x34b6[283]];
    this[_0x34b6[318]][_0x34b6[267]][_0x34b6[284]][_0x34b6[275]] = this[_0x34b6[284]];
    if (local$$5432) {
      local$$5428[_0x34b6[227]][_0x34b6[359]](local$$5428[_0x34b6[227]].STENCIL_TEST);
    }
    local$$5428[_0x34b6[225]](this[_0x34b6[239]], this[_0x34b6[240]], local$$5430, false);
    local$$5428[_0x34b6[262]](this[_0x34b6[324]], this[_0x34b6[325]]);
    local$$5428[_0x34b6[258]] = local$$5470;
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
      vertexShader : _0x34b6[360],
      fragmentShader : _0x34b6[361]
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
      vertexShader : _0x34b6[362],
      fragmentShader : _0x34b6[363]
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
      vertexShader : _0x34b6[362],
      fragmentShader : _0x34b6[364]
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
      vertexShader : _0x34b6[362],
      fragmentShader : _0x34b6[365],
      blending : THREE[_0x34b6[366]],
      depthTest : false,
      depthWrite : false,
      transparent : true
    });
  }
});
THREE[_0x34b6[277]][_0x34b6[354]] = new THREE.Vector2(1, 0);
THREE[_0x34b6[277]][_0x34b6[355]] = new THREE.Vector2(0, 1);
!function(local$$6495) {
  if (_0x34b6[368] == typeof exports && _0x34b6[367] != typeof module) {
    module[_0x34b6[369]] = local$$6495();
  } else {
    if (_0x34b6[391] == typeof define && define[_0x34b6[392]]) {
      define([], local$$6495);
    } else {
      var local$$6528;
      if (_0x34b6[367] != typeof window) {
        /** @type {!Window} */
        local$$6528 = window;
      } else {
        if (_0x34b6[367] != typeof global) {
          local$$6528 = global;
        } else {
          if (_0x34b6[367] != typeof self) {
            /** @type {!Window} */
            local$$6528 = self;
          }
        }
      }
      local$$6528[_0x34b6[518]] = local$$6495();
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
          var local$$6597 = typeof require == _0x34b6[391] && require;
          if (!local$$6585 && local$$6597) {
            return local$$6597(local$$6584, true);
          }
          if (local$$6607) {
            return local$$6607(local$$6584, true);
          }
          /** @type {!Error} */
          var local$$6622 = new Error(_0x34b6[1050] + local$$6584 + _0x34b6[1051]);
          throw local$$6622[_0x34b6[1052]] = _0x34b6[1053], local$$6622;
        }
        var local$$6639 = local$$6580[local$$6584] = {
          exports : {}
        };
        local$$6579[local$$6584][0][_0x34b6[238]](local$$6639[_0x34b6[369]], function(local$$6650) {
          var local$$6656 = local$$6579[local$$6584][1][local$$6650];
          return local$$6583(local$$6656 ? local$$6656 : local$$6650);
        }, local$$6639, local$$6639[_0x34b6[369]], local$$6578, local$$6579, local$$6580, local$$6581);
      }
      return local$$6580[local$$6584][_0x34b6[369]];
    }
    var local$$6607 = typeof require == _0x34b6[391] && require;
    /** @type {number} */
    var local$$6685 = 0;
    for (; local$$6685 < local$$6581[_0x34b6[223]]; local$$6685++) {
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
            var local$$6729 = local$$6723[_0x34b6[223]];
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
            return local$$6722(local$$6746[_0x34b6[379]](local$$6752), local$$6747)[_0x34b6[2]](_0x34b6[378]);
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
            var local$$6776 = local$$6765[_0x34b6[223]];
            var local$$6778;
            var local$$6780;
            for (; local$$6771 < local$$6776;) {
              local$$6778 = local$$6765[_0x34b6[380]](local$$6771++);
              if (local$$6778 >= 55296 && local$$6778 <= 56319 && local$$6771 < local$$6776) {
                local$$6780 = local$$6765[_0x34b6[380]](local$$6771++);
                if ((local$$6780 & 64512) == 56320) {
                  local$$6768[_0x34b6[220]](((local$$6778 & 1023) << 10) + (local$$6780 & 1023) + 65536);
                } else {
                  local$$6768[_0x34b6[220]](local$$6778);
                  local$$6771--;
                }
              } else {
                local$$6768[_0x34b6[220]](local$$6778);
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
              var local$$6855 = _0x34b6[381];
              if (local$$6851 > 65535) {
                /** @type {number} */
                local$$6851 = local$$6851 - 65536;
                local$$6855 = local$$6855 + local$$6863(local$$6851 >>> 10 & 1023 | 55296);
                /** @type {number} */
                local$$6851 = 56320 | local$$6851 & 1023;
              }
              local$$6855 = local$$6855 + local$$6863(local$$6851);
              return local$$6855;
            })[_0x34b6[2]](_0x34b6[381]);
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
            var local$$7014 = local$$7006[_0x34b6[223]];
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
            local$$7027 = local$$7006[_0x34b6[382]](local$$7048);
            if (local$$7027 < 0) {
              /** @type {number} */
              local$$7027 = 0;
            }
            /** @type {number} */
            local$$7029 = 0;
            for (; local$$7029 < local$$7027; ++local$$7029) {
              if (local$$7006[_0x34b6[380]](local$$7029) >= 128) {
                local$$6712(_0x34b6[383]);
              }
              local$$7009[_0x34b6[220]](local$$7006[_0x34b6[380]](local$$7029));
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
                  local$$6712(_0x34b6[384]);
                }
                local$$7039 = local$$6900(local$$7006[_0x34b6[380]](local$$7031++));
                if (local$$7039 >= local$$6933 || local$$7039 > local$$6964((local$$7130 - local$$7019) / local$$7035)) {
                  local$$6712(_0x34b6[385]);
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
                  local$$6712(_0x34b6[385]);
                }
                /** @type {number} */
                local$$7035 = local$$7035 * local$$7043;
              }
              local$$7016 = local$$7009[_0x34b6[223]] + 1;
              local$$7025 = local$$6956(local$$7019 - local$$7033, local$$7016, local$$7033 == 0);
              if (local$$6964(local$$7019 / local$$7016) > local$$7130 - local$$7022) {
                local$$6712(_0x34b6[385]);
              }
              local$$7022 = local$$7022 + local$$6964(local$$7019 / local$$7016);
              /** @type {number} */
              local$$7019 = local$$7019 % local$$7016;
              local$$7009[_0x34b6[222]](local$$7019++, 0, local$$7022);
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
            local$$7254 = local$$7227[_0x34b6[223]];
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
                local$$7252[_0x34b6[220]](local$$6863(local$$7249));
              }
            }
            local$$7233 = local$$7235 = local$$7252[_0x34b6[223]];
            if (local$$7235) {
              local$$7252[_0x34b6[220]](local$$7048);
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
                local$$6712(_0x34b6[385]);
              }
              /** @type {number} */
              local$$7231 = local$$7231 + (local$$7241 - local$$7229) * local$$7256;
              local$$7229 = local$$7241;
              /** @type {number} */
              local$$7239 = 0;
              for (; local$$7239 < local$$7254; ++local$$7239) {
                local$$7249 = local$$7227[local$$7239];
                if (local$$7249 < local$$7229 && ++local$$7231 > local$$7130) {
                  local$$6712(_0x34b6[385]);
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
                    local$$7252[_0x34b6[220]](local$$6863(local$$6937(local$$7247 + local$$7260 % local$$7258, 0)));
                    local$$7243 = local$$6964(local$$7260 / local$$7258);
                  }
                  local$$7252[_0x34b6[220]](local$$6863(local$$6937(local$$7243, 0)));
                  local$$7237 = local$$6956(local$$7231, local$$7256, local$$7233 == local$$7235);
                  /** @type {number} */
                  local$$7231 = 0;
                  ++local$$7233;
                }
              }
              ++local$$7231;
              ++local$$7229;
            }
            return local$$7252[_0x34b6[2]](_0x34b6[381]);
          }
          /**
           * @param {?} local$$7464
           * @return {?}
           */
          function local$$7463(local$$7464) {
            return local$$6745(local$$7464, function(local$$7466) {
              return local$$7468[_0x34b6[386]](local$$7466) ? local$$7005(local$$7466[_0x34b6[388]](4)[_0x34b6[387]]()) : local$$7466;
            });
          }
          /**
           * @param {?} local$$7492
           * @return {?}
           */
          function local$$7491(local$$7492) {
            return local$$6745(local$$7492, function(local$$7494) {
              return local$$7496[_0x34b6[386]](local$$7494) ? _0x34b6[389] + local$$7226(local$$7494) : local$$7494;
            });
          }
          var local$$7518 = typeof local$$6704 == _0x34b6[368] && local$$6704;
          var local$$7531 = typeof local$$6703 == _0x34b6[368] && local$$6703 && local$$6703[_0x34b6[369]] == local$$7518 && local$$6703;
          var local$$7538 = typeof local$$6706 == _0x34b6[368] && local$$6706;
          if (local$$7538[_0x34b6[370]] === local$$7538 || local$$7538[_0x34b6[371]] === local$$7538) {
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
          var local$$7048 = _0x34b6[372];
          /** @type {!RegExp} */
          var local$$7468 = /^xn--/;
          /** @type {!RegExp} */
          var local$$7496 = /[^ -~]/;
          /** @type {!RegExp} */
          var local$$6752 = /\x2E|\u3002|\uFF0E|\uFF61/g;
          var local$$6716 = {
            "overflow" : _0x34b6[373],
            "not-basic" : _0x34b6[374],
            "invalid-input" : _0x34b6[375]
          };
          /** @type {number} */
          var local$$6979 = local$$6933 - local$$7148;
          var local$$6964 = Math[_0x34b6[376]];
          var local$$6863 = String[_0x34b6[377]];
          var local$$7603;
          local$$7554 = {
            "version" : _0x34b6[390],
            "ucs2" : {
              "decode" : local$$6764,
              "encode" : local$$6848
            },
            "decode" : local$$7005,
            "encode" : local$$7226,
            "toASCII" : local$$7491,
            "toUnicode" : local$$7463
          };
          if (typeof local$$6572 == _0x34b6[391] && typeof local$$6572[_0x34b6[392]] == _0x34b6[368] && local$$6572[_0x34b6[392]]) {
            local$$6572(_0x34b6[393], function() {
              return local$$7554;
            });
          } else {
            if (local$$7518 && !local$$7518[_0x34b6[394]]) {
              if (local$$7531) {
                local$$7531[_0x34b6[369]] = local$$7554;
              } else {
                for (local$$7603 in local$$7554) {
                  if (local$$7554[_0x34b6[395]](local$$7603)) {
                    local$$7518[local$$7603] = local$$7554[local$$7603];
                  }
                }
              }
            } else {
              local$$6710[_0x34b6[393]] = local$$7554;
            }
          }
        })(this);
      })[_0x34b6[238]](this, typeof global !== _0x34b6[367] ? global : typeof self !== _0x34b6[367] ? self : typeof window !== _0x34b6[367] ? window : {});
    }, {}],
    2 : [function(local$$7705, local$$7706, local$$7707) {
      /**
       * @param {?} local$$7710
       * @param {?} local$$7711
       * @param {?} local$$7712
       * @return {undefined}
       */
      function local$$7709(local$$7710, local$$7711, local$$7712) {
        if (local$$7710[_0x34b6[397]] && (local$$7711 !== local$$7710[_0x34b6[397]][_0x34b6[398]] || local$$7712 !== local$$7710[_0x34b6[397]][_0x34b6[399]])) {
          local$$7710[_0x34b6[397]][_0x34b6[400]](local$$7711, local$$7712);
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
            local$$7747[_0x34b6[208]] = local$$7746[_0x34b6[208]];
            local$$7747[_0x34b6[209]] = local$$7746[_0x34b6[209]];
            local$$7747[_0x34b6[403]](_0x34b6[402])[_0x34b6[404]](local$$7746[_0x34b6[403]](_0x34b6[402])[_0x34b6[401]](0, 0, local$$7746[_0x34b6[208]], local$$7746[_0x34b6[209]]), 0, 0);
          }
        } catch (local$$7799) {
          local$$7800(_0x34b6[405], local$$7746, local$$7799);
        }
      }
      /**
       * @param {?} local$$7812
       * @param {boolean} local$$7813
       * @return {?}
       */
      function local$$7811(local$$7812, local$$7813) {
        var local$$7834 = local$$7812[_0x34b6[394]] === 3 ? document[_0x34b6[407]](local$$7812[_0x34b6[406]]) : local$$7812[_0x34b6[408]](false);
        var local$$7839 = local$$7812[_0x34b6[409]];
        for (; local$$7839;) {
          if (local$$7813 === true || local$$7839[_0x34b6[394]] !== 1 || local$$7839[_0x34b6[410]] !== _0x34b6[411]) {
            local$$7834[_0x34b6[412]](local$$7811(local$$7839, local$$7813));
          }
          local$$7839 = local$$7839[_0x34b6[413]];
        }
        if (local$$7812[_0x34b6[394]] === 1) {
          local$$7834[_0x34b6[414]] = local$$7812[_0x34b6[415]];
          local$$7834[_0x34b6[416]] = local$$7812[_0x34b6[417]];
          if (local$$7812[_0x34b6[410]] === _0x34b6[418]) {
            local$$7745(local$$7812, local$$7834);
          } else {
            if (local$$7812[_0x34b6[410]] === _0x34b6[419] || local$$7812[_0x34b6[410]] === _0x34b6[420]) {
              local$$7834[_0x34b6[275]] = local$$7812[_0x34b6[275]];
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
        if (local$$7937[_0x34b6[394]] === 1) {
          local$$7937[_0x34b6[415]] = local$$7937[_0x34b6[414]];
          local$$7937[_0x34b6[417]] = local$$7937[_0x34b6[416]];
          var local$$7963 = local$$7937[_0x34b6[409]];
          for (; local$$7963;) {
            local$$7936(local$$7963);
            local$$7963 = local$$7963[_0x34b6[413]];
          }
        }
      }
      var local$$7800 = local$$7705(_0x34b6[396]);
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
      local$$7706[_0x34b6[369]] = function(local$$7987, local$$7988, local$$7989, local$$7990, local$$7991, local$$7992, local$$7993) {
        var local$$8002 = local$$7811(local$$7987[_0x34b6[421]], local$$7991[_0x34b6[422]]);
        var local$$8010 = local$$7988[_0x34b6[424]](_0x34b6[423]);
        local$$8010[_0x34b6[425]] = _0x34b6[426];
        local$$8010[_0x34b6[428]][_0x34b6[427]] = _0x34b6[429];
        local$$8010[_0x34b6[428]][_0x34b6[430]] = _0x34b6[431];
        local$$8010[_0x34b6[428]][_0x34b6[432]] = _0x34b6[433];
        local$$8010[_0x34b6[428]][_0x34b6[434]] = _0x34b6[435];
        local$$8010[_0x34b6[428]][_0x34b6[436]] = _0x34b6[437];
        local$$8010[_0x34b6[208]] = local$$7989;
        local$$8010[_0x34b6[209]] = local$$7990;
        local$$8010[_0x34b6[438]] = _0x34b6[439];
        local$$7988[_0x34b6[440]][_0x34b6[412]](local$$8010);
        return new Promise(function(local$$8095) {
          var local$$8103 = local$$8010[_0x34b6[442]][_0x34b6[441]];
          /** @type {function(): undefined} */
          local$$8010[_0x34b6[442]][_0x34b6[443]] = local$$8010[_0x34b6[443]] = function() {
            /** @type {number} */
            var local$$8134 = setInterval(function() {
              if (local$$8103[_0x34b6[440]][_0x34b6[444]][_0x34b6[223]] > 0) {
                local$$7936(local$$8103[_0x34b6[421]]);
                clearInterval(local$$8134);
                if (local$$7991[_0x34b6[445]] === _0x34b6[446]) {
                  local$$8010[_0x34b6[442]][_0x34b6[400]](local$$7992, local$$7993);
                  if (/(iPad|iPhone|iPod)/g[_0x34b6[386]](navigator[_0x34b6[447]]) && (local$$8010[_0x34b6[442]][_0x34b6[448]] !== local$$7993 || local$$8010[_0x34b6[442]][_0x34b6[449]] !== local$$7992)) {
                    local$$8103[_0x34b6[421]][_0x34b6[428]][_0x34b6[434]] = -local$$7993 + _0x34b6[450];
                    local$$8103[_0x34b6[421]][_0x34b6[428]][_0x34b6[432]] = -local$$7992 + _0x34b6[450];
                    local$$8103[_0x34b6[421]][_0x34b6[428]][_0x34b6[430]] = _0x34b6[451];
                  }
                }
                local$$8095(local$$8010);
              }
            }, 50);
          };
          local$$8103[_0x34b6[452]]();
          local$$8103[_0x34b6[454]](_0x34b6[453]);
          local$$7709(local$$7987, local$$7992, local$$7993);
          local$$8103[_0x34b6[456]](local$$8103[_0x34b6[455]](local$$8002), local$$8103[_0x34b6[421]]);
          local$$8103[_0x34b6[457]]();
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
        this[_0x34b6[458]] = 0;
        /** @type {number} */
        this[_0x34b6[459]] = 0;
        /** @type {number} */
        this[_0x34b6[460]] = 0;
        /** @type {null} */
        this[_0x34b6[461]] = null;
        var local$$8344 = this[_0x34b6[462]](local$$8289) || this[_0x34b6[463]](local$$8289) || this[_0x34b6[464]](local$$8289) || this[_0x34b6[465]](local$$8289) || this[_0x34b6[466]](local$$8289) || this[_0x34b6[467]](local$$8289);
      }
      /**
       * @param {number} local$$8354
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[468]] = function(local$$8354) {
        /** @type {number} */
        var local$$8358 = 1 - local$$8354;
        return new local$$8288([Math[_0x34b6[292]](this[_0x34b6[458]] * local$$8358), Math[_0x34b6[292]](this[_0x34b6[459]] * local$$8358), Math[_0x34b6[292]](this[_0x34b6[460]] * local$$8358), this[_0x34b6[461]]]);
      };
      /**
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[469]] = function() {
        return this[_0x34b6[461]] === 0;
      };
      /**
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[470]] = function() {
        return this[_0x34b6[458]] === 0 && this[_0x34b6[459]] === 0 && this[_0x34b6[460]] === 0;
      };
      /**
       * @param {!Array} local$$8446
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[462]] = function(local$$8446) {
        if (Array[_0x34b6[471]](local$$8446)) {
          this[_0x34b6[458]] = Math[_0x34b6[472]](local$$8446[0], 255);
          this[_0x34b6[459]] = Math[_0x34b6[472]](local$$8446[1], 255);
          this[_0x34b6[460]] = Math[_0x34b6[472]](local$$8446[2], 255);
          if (local$$8446[_0x34b6[223]] > 3) {
            this[_0x34b6[461]] = local$$8446[3];
          }
        }
        return Array[_0x34b6[471]](local$$8446);
      };
      /** @type {!RegExp} */
      var local$$8518 = /^#([a-f0-9]{3})$/i;
      /**
       * @param {?} local$$8526
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[467]] = function(local$$8526) {
        /** @type {null} */
        var local$$8529 = null;
        if ((local$$8529 = local$$8526[_0x34b6[473]](local$$8518)) !== null) {
          /** @type {number} */
          this[_0x34b6[458]] = parseInt(local$$8529[1][0] + local$$8529[1][0], 16);
          /** @type {number} */
          this[_0x34b6[459]] = parseInt(local$$8529[1][1] + local$$8529[1][1], 16);
          /** @type {number} */
          this[_0x34b6[460]] = parseInt(local$$8529[1][2] + local$$8529[1][2], 16);
        }
        return local$$8529 !== null;
      };
      /** @type {!RegExp} */
      var local$$8599 = /^#([a-f0-9]{6})$/i;
      /**
       * @param {?} local$$8607
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[466]] = function(local$$8607) {
        /** @type {null} */
        var local$$8610 = null;
        if ((local$$8610 = local$$8607[_0x34b6[473]](local$$8599)) !== null) {
          /** @type {number} */
          this[_0x34b6[458]] = parseInt(local$$8610[1][_0x34b6[474]](0, 2), 16);
          /** @type {number} */
          this[_0x34b6[459]] = parseInt(local$$8610[1][_0x34b6[474]](2, 4), 16);
          /** @type {number} */
          this[_0x34b6[460]] = parseInt(local$$8610[1][_0x34b6[474]](4, 6), 16);
        }
        return local$$8610 !== null;
      };
      /** @type {!RegExp} */
      var local$$8676 = /^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$/;
      /**
       * @param {?} local$$8684
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[464]] = function(local$$8684) {
        /** @type {null} */
        var local$$8687 = null;
        if ((local$$8687 = local$$8684[_0x34b6[473]](local$$8676)) !== null) {
          /** @type {number} */
          this[_0x34b6[458]] = Number(local$$8687[1]);
          /** @type {number} */
          this[_0x34b6[459]] = Number(local$$8687[2]);
          /** @type {number} */
          this[_0x34b6[460]] = Number(local$$8687[3]);
        }
        return local$$8687 !== null;
      };
      /** @type {!RegExp} */
      var local$$8733 = /^rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d?\.?\d+)\s*\)$/;
      /**
       * @param {?} local$$8741
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[465]] = function(local$$8741) {
        /** @type {null} */
        var local$$8744 = null;
        if ((local$$8744 = local$$8741[_0x34b6[473]](local$$8733)) !== null) {
          /** @type {number} */
          this[_0x34b6[458]] = Number(local$$8744[1]);
          /** @type {number} */
          this[_0x34b6[459]] = Number(local$$8744[2]);
          /** @type {number} */
          this[_0x34b6[460]] = Number(local$$8744[3]);
          /** @type {number} */
          this[_0x34b6[461]] = Number(local$$8744[4]);
        }
        return local$$8744 !== null;
      };
      /**
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[475]] = function() {
        return this[_0x34b6[461]] !== null && this[_0x34b6[461]] !== 1 ? _0x34b6[476] + [this[_0x34b6[458]], this[_0x34b6[459]], this[_0x34b6[460]], this[_0x34b6[461]]][_0x34b6[2]](_0x34b6[477]) + _0x34b6[478] : _0x34b6[479] + [this[_0x34b6[458]], this[_0x34b6[459]], this[_0x34b6[460]]][_0x34b6[2]](_0x34b6[477]) + _0x34b6[478];
      };
      /**
       * @param {?} local$$8872
       * @return {?}
       */
      local$$8288[_0x34b6[219]][_0x34b6[463]] = function(local$$8872) {
        local$$8872 = local$$8872[_0x34b6[387]]();
        var local$$8882 = local$$8880[local$$8872];
        if (local$$8882) {
          this[_0x34b6[458]] = local$$8882[0];
          this[_0x34b6[459]] = local$$8882[1];
          this[_0x34b6[460]] = local$$8882[2];
        } else {
          if (local$$8872 === _0x34b6[480]) {
            /** @type {number} */
            this[_0x34b6[458]] = this[_0x34b6[459]] = this[_0x34b6[460]] = this[_0x34b6[461]] = 0;
            return true;
          }
        }
        return !!local$$8882;
      };
      /** @type {boolean} */
      local$$8288[_0x34b6[219]][_0x34b6[481]] = true;
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
      local$$8285[_0x34b6[369]] = local$$8288;
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
        if (local$$9705[_0x34b6[493]]) {
          /** @type {boolean} */
          local$$9718[_0x34b6[494]][_0x34b6[493]] = true;
          local$$9718[_0x34b6[494]][_0x34b6[495]] = Date[_0x34b6[348]]();
        }
        local$$9705[_0x34b6[496]] = typeof local$$9705[_0x34b6[496]] === _0x34b6[367] ? true : local$$9705[_0x34b6[496]];
        local$$9705[_0x34b6[497]] = typeof local$$9705[_0x34b6[497]] === _0x34b6[367] ? false : local$$9705[_0x34b6[497]];
        local$$9705[_0x34b6[498]] = typeof local$$9705[_0x34b6[498]] === _0x34b6[367] ? true : local$$9705[_0x34b6[498]];
        local$$9705[_0x34b6[422]] = typeof local$$9705[_0x34b6[422]] === _0x34b6[367] ? false : local$$9705[_0x34b6[422]];
        local$$9705[_0x34b6[499]] = typeof local$$9705[_0x34b6[499]] === _0x34b6[367] ? 1E4 : local$$9705[_0x34b6[499]];
        local$$9705[_0x34b6[204]] = typeof local$$9705[_0x34b6[204]] === _0x34b6[391] ? local$$9705[_0x34b6[204]] : local$$9842;
        /** @type {boolean} */
        local$$9705[_0x34b6[500]] = !!local$$9705[_0x34b6[500]];
        if (typeof local$$9704 === _0x34b6[501]) {
          if (typeof local$$9705[_0x34b6[502]] !== _0x34b6[501]) {
            return Promise[_0x34b6[504]](_0x34b6[503]);
          }
          var local$$9889 = local$$9705[_0x34b6[208]] != null ? local$$9705[_0x34b6[208]] : window[_0x34b6[505]];
          var local$$9903 = local$$9705[_0x34b6[209]] != null ? local$$9705[_0x34b6[209]] : window[_0x34b6[506]];
          return local$$9905(local$$9906(local$$9704), local$$9705[_0x34b6[502]], document, local$$9889, local$$9903, local$$9705)[_0x34b6[507]](function(local$$9915) {
            return local$$9917(local$$9915[_0x34b6[442]][_0x34b6[441]][_0x34b6[421]], local$$9915, local$$9705, local$$9889, local$$9903);
          });
        }
        var local$$9949 = (local$$9704 === undefined ? [document[_0x34b6[421]]] : local$$9704[_0x34b6[223]] ? local$$9704 : [local$$9704])[0];
        local$$9949[_0x34b6[508]](local$$9954 + local$$9709, local$$9709);
        return local$$9958(local$$9949[_0x34b6[511]], local$$9705, local$$9949[_0x34b6[511]][_0x34b6[397]][_0x34b6[505]], local$$9949[_0x34b6[511]][_0x34b6[397]][_0x34b6[506]], local$$9709)[_0x34b6[507]](function(local$$9984) {
          if (typeof local$$9705[_0x34b6[509]] === _0x34b6[391]) {
            local$$9718(_0x34b6[510]);
            local$$9705[_0x34b6[509]](local$$9984);
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
        return local$$10018(local$$10012, local$$10012, local$$10014, local$$10015, local$$10013, local$$10012[_0x34b6[397]][_0x34b6[398]], local$$10012[_0x34b6[397]][_0x34b6[399]])[_0x34b6[507]](function(local$$10035) {
          local$$9718(_0x34b6[519]);
          var local$$10042 = local$$9954 + local$$10016;
          var local$$10054 = _0x34b6[520] + local$$10042 + _0x34b6[521] + local$$10016 + _0x34b6[522];
          local$$10012[_0x34b6[524]](local$$10054)[_0x34b6[523]](local$$10042);
          var local$$10068 = local$$10035[_0x34b6[442]];
          var local$$10077 = local$$10068[_0x34b6[441]][_0x34b6[524]](local$$10054);
          var local$$10103 = typeof local$$10013[_0x34b6[525]] === _0x34b6[391] ? Promise[_0x34b6[526]](local$$10013[_0x34b6[525]](local$$10068[_0x34b6[441]])) : Promise[_0x34b6[526]](true);
          return local$$10103[_0x34b6[507]](function() {
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
        var local$$10130 = local$$10122[_0x34b6[442]];
        var local$$10137 = new local$$10132(local$$10130[_0x34b6[441]]);
        var local$$10141 = new local$$10139(local$$10123, local$$10137);
        var local$$10145 = local$$10143(local$$10121);
        var local$$10159 = local$$10123[_0x34b6[445]] === _0x34b6[446] ? local$$10124 : local$$10153(local$$10130[_0x34b6[441]]);
        var local$$10173 = local$$10123[_0x34b6[445]] === _0x34b6[446] ? local$$10125 : local$$10167(local$$10130[_0x34b6[441]]);
        var local$$10179 = new local$$10123[_0x34b6[204]](local$$10159, local$$10173, local$$10141, local$$10123, document);
        var local$$10183 = new local$$10181(local$$10121, local$$10179, local$$10137, local$$10141, local$$10123);
        return local$$10183[_0x34b6[528]][_0x34b6[507]](function() {
          local$$9718(_0x34b6[527]);
          var local$$10196;
          if (local$$10123[_0x34b6[445]] === _0x34b6[446]) {
            local$$10196 = local$$10204(local$$10179[_0x34b6[516]], {
              width : local$$10179[_0x34b6[516]][_0x34b6[208]],
              height : local$$10179[_0x34b6[516]][_0x34b6[209]],
              top : 0,
              left : 0,
              x : 0,
              y : 0
            });
          } else {
            if (local$$10121 === local$$10130[_0x34b6[441]][_0x34b6[440]] || local$$10121 === local$$10130[_0x34b6[441]][_0x34b6[421]] || local$$10123[_0x34b6[516]] != null) {
              local$$10196 = local$$10179[_0x34b6[516]];
            } else {
              local$$10196 = local$$10204(local$$10179[_0x34b6[516]], {
                width : local$$10123[_0x34b6[208]] != null ? local$$10123[_0x34b6[208]] : local$$10145[_0x34b6[208]],
                height : local$$10123[_0x34b6[209]] != null ? local$$10123[_0x34b6[209]] : local$$10145[_0x34b6[209]],
                top : local$$10145[_0x34b6[434]],
                left : local$$10145[_0x34b6[432]],
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
        if (local$$10311[_0x34b6[498]]) {
          local$$10310[_0x34b6[530]][_0x34b6[529]](local$$10310);
          local$$9718(_0x34b6[531]);
        }
      }
      /**
       * @param {?} local$$10332
       * @param {?} local$$10333
       * @return {?}
       */
      function local$$10204(local$$10332, local$$10333) {
        var local$$10341 = document[_0x34b6[424]](_0x34b6[516]);
        var local$$10360 = Math[_0x34b6[472]](local$$10332[_0x34b6[208]] - 1, Math[_0x34b6[532]](0, local$$10333[_0x34b6[432]]));
        var local$$10381 = Math[_0x34b6[472]](local$$10332[_0x34b6[208]], Math[_0x34b6[532]](1, local$$10333[_0x34b6[432]] + local$$10333[_0x34b6[208]]));
        var local$$10400 = Math[_0x34b6[472]](local$$10332[_0x34b6[209]] - 1, Math[_0x34b6[532]](0, local$$10333[_0x34b6[434]]));
        var local$$10421 = Math[_0x34b6[472]](local$$10332[_0x34b6[209]], Math[_0x34b6[532]](1, local$$10333[_0x34b6[434]] + local$$10333[_0x34b6[209]]));
        local$$10341[_0x34b6[208]] = local$$10333[_0x34b6[208]];
        local$$10341[_0x34b6[209]] = local$$10333[_0x34b6[209]];
        /** @type {number} */
        var local$$10440 = local$$10381 - local$$10360;
        /** @type {number} */
        var local$$10443 = local$$10421 - local$$10400;
        local$$9718(_0x34b6[533], _0x34b6[534], local$$10333[_0x34b6[432]], _0x34b6[535], local$$10333[_0x34b6[434]], _0x34b6[536], local$$10440, _0x34b6[537], local$$10443);
        local$$9718(_0x34b6[538], local$$10333[_0x34b6[208]], _0x34b6[539], local$$10333[_0x34b6[209]], _0x34b6[540], local$$10360, _0x34b6[541], local$$10400);
        local$$10341[_0x34b6[403]](_0x34b6[402])[_0x34b6[542]](local$$10332, local$$10360, local$$10400, local$$10440, local$$10443, local$$10333[_0x34b6[290]], local$$10333[_0x34b6[291]], local$$10440, local$$10443);
        return local$$10341;
      }
      /**
       * @param {?} local$$10499
       * @return {?}
       */
      function local$$10153(local$$10499) {
        return Math[_0x34b6[532]](Math[_0x34b6[532]](local$$10499[_0x34b6[440]][_0x34b6[543]], local$$10499[_0x34b6[421]][_0x34b6[543]]), Math[_0x34b6[532]](local$$10499[_0x34b6[440]][_0x34b6[544]], local$$10499[_0x34b6[421]][_0x34b6[544]]), Math[_0x34b6[532]](local$$10499[_0x34b6[440]][_0x34b6[545]], local$$10499[_0x34b6[421]][_0x34b6[545]]));
      }
      /**
       * @param {?} local$$10556
       * @return {?}
       */
      function local$$10167(local$$10556) {
        return Math[_0x34b6[532]](Math[_0x34b6[532]](local$$10556[_0x34b6[440]][_0x34b6[546]], local$$10556[_0x34b6[421]][_0x34b6[546]]), Math[_0x34b6[532]](local$$10556[_0x34b6[440]][_0x34b6[547]], local$$10556[_0x34b6[421]][_0x34b6[547]]), Math[_0x34b6[532]](local$$10556[_0x34b6[440]][_0x34b6[548]], local$$10556[_0x34b6[421]][_0x34b6[548]]));
      }
      /**
       * @param {string} local$$10613
       * @return {?}
       */
      function local$$9906(local$$10613) {
        var local$$10621 = document[_0x34b6[424]](_0x34b6[461]);
        /** @type {string} */
        local$$10621[_0x34b6[549]] = local$$10613;
        local$$10621[_0x34b6[549]] = local$$10621[_0x34b6[549]];
        return local$$10621;
      }
      var local$$10132 = local$$9699(_0x34b6[482]);
      var local$$9842 = local$$9699(_0x34b6[483]);
      var local$$10139 = local$$9699(_0x34b6[484]);
      var local$$10181 = local$$9699(_0x34b6[485]);
      var local$$10658 = local$$9699(_0x34b6[486]);
      var local$$9718 = local$$9699(_0x34b6[396]);
      var local$$10667 = local$$9699(_0x34b6[487]);
      var local$$10018 = local$$9699(_0x34b6[488]);
      var local$$9905 = local$$9699(_0x34b6[490])[_0x34b6[489]];
      var local$$10143 = local$$10667[_0x34b6[491]];
      var local$$9954 = _0x34b6[492];
      /** @type {number} */
      var local$$9707 = 0;
      local$$9703[_0x34b6[512]] = local$$9842;
      local$$9703[_0x34b6[513]] = local$$10658;
      local$$9703[_0x34b6[514]] = local$$9718;
      local$$9703[_0x34b6[515]] = local$$10667;
      /** @type {!Function} */
      var local$$10746 = typeof document === _0x34b6[367] || typeof Object[_0x34b6[242]] !== _0x34b6[391] || typeof document[_0x34b6[424]](_0x34b6[516])[_0x34b6[403]] !== _0x34b6[391] ? function() {
        return Promise[_0x34b6[504]](_0x34b6[517]);
      } : local$$9703;
      /** @type {!Function} */
      local$$9700[_0x34b6[369]] = local$$10746;
      if (typeof local$$6572 === _0x34b6[391] && local$$6572[_0x34b6[392]]) {
        local$$6572(_0x34b6[518], [], function() {
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
        this[_0x34b6[551]] = local$$10792;
        local$$10799(_0x34b6[552], local$$10792);
        if (!this[_0x34b6[553]] || !this[_0x34b6[554]]) {
          local$$10799(_0x34b6[555]);
          /** @type {!Image} */
          local$$10791[_0x34b6[219]][_0x34b6[554]] = new Image;
          var local$$10830 = this[_0x34b6[554]];
          /** @type {!Promise} */
          local$$10791[_0x34b6[219]][_0x34b6[553]] = new Promise(function(local$$10838, local$$10839) {
            local$$10830[_0x34b6[443]] = local$$10838;
            local$$10830[_0x34b6[556]] = local$$10839;
            local$$10830[_0x34b6[551]] = local$$10854();
            if (local$$10830[_0x34b6[557]] === true) {
              local$$10838(local$$10830);
            }
          });
        }
      }
      var local$$10799 = local$$10787(_0x34b6[396]);
      var local$$10854 = local$$10787(_0x34b6[487])[_0x34b6[550]];
      /** @type {function(?): undefined} */
      local$$10788[_0x34b6[369]] = local$$10791;
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
        var local$$10913 = document[_0x34b6[424]](_0x34b6[558]);
        var local$$10921 = document[_0x34b6[424]](_0x34b6[559]);
        var local$$10929 = document[_0x34b6[424]](_0x34b6[560]);
        var local$$10933 = _0x34b6[561];
        var local$$10935;
        var local$$10937;
        local$$10913[_0x34b6[428]][_0x34b6[427]] = _0x34b6[429];
        local$$10913[_0x34b6[428]][_0x34b6[562]] = local$$10904;
        local$$10913[_0x34b6[428]][_0x34b6[563]] = local$$10905;
        /** @type {number} */
        local$$10913[_0x34b6[428]][_0x34b6[564]] = 0;
        /** @type {number} */
        local$$10913[_0x34b6[428]][_0x34b6[565]] = 0;
        document[_0x34b6[440]][_0x34b6[412]](local$$10913);
        local$$10921[_0x34b6[551]] = local$$10994();
        /** @type {number} */
        local$$10921[_0x34b6[208]] = 1;
        /** @type {number} */
        local$$10921[_0x34b6[209]] = 1;
        /** @type {number} */
        local$$10921[_0x34b6[428]][_0x34b6[564]] = 0;
        /** @type {number} */
        local$$10921[_0x34b6[428]][_0x34b6[565]] = 0;
        local$$10921[_0x34b6[428]][_0x34b6[566]] = _0x34b6[567];
        local$$10929[_0x34b6[428]][_0x34b6[562]] = local$$10904;
        local$$10929[_0x34b6[428]][_0x34b6[563]] = local$$10905;
        /** @type {number} */
        local$$10929[_0x34b6[428]][_0x34b6[564]] = 0;
        /** @type {number} */
        local$$10929[_0x34b6[428]][_0x34b6[565]] = 0;
        local$$10929[_0x34b6[412]](document[_0x34b6[407]](local$$10933));
        local$$10913[_0x34b6[412]](local$$10929);
        local$$10913[_0x34b6[412]](local$$10921);
        /** @type {number} */
        local$$10935 = local$$10921[_0x34b6[568]] - local$$10929[_0x34b6[568]] + 1;
        local$$10913[_0x34b6[529]](local$$10929);
        local$$10913[_0x34b6[412]](document[_0x34b6[407]](local$$10933));
        local$$10913[_0x34b6[428]][_0x34b6[569]] = _0x34b6[570];
        local$$10921[_0x34b6[428]][_0x34b6[566]] = _0x34b6[571];
        /** @type {number} */
        local$$10937 = local$$10921[_0x34b6[568]] - local$$10913[_0x34b6[568]] + 1;
        document[_0x34b6[440]][_0x34b6[529]](local$$10913);
        /** @type {number} */
        this[_0x34b6[567]] = local$$10935;
        /** @type {number} */
        this[_0x34b6[572]] = 1;
        /** @type {number} */
        this[_0x34b6[573]] = local$$10937;
      }
      var local$$10994 = local$$10899(_0x34b6[487])[_0x34b6[550]];
      /** @type {function(?, ?): undefined} */
      local$$10900[_0x34b6[369]] = local$$10903;
    }, {
      "./utils" : 26
    }],
    7 : [function(local$$11191, local$$11192, local$$11193) {
      /**
       * @return {undefined}
       */
      function local$$11195() {
        this[_0x34b6[575]] = {};
      }
      var local$$11208 = local$$11191(_0x34b6[574]);
      /**
       * @param {?} local$$11216
       * @param {?} local$$11217
       * @return {?}
       */
      local$$11195[_0x34b6[219]][_0x34b6[576]] = function(local$$11216, local$$11217) {
        if (this[_0x34b6[575]][local$$11216 + _0x34b6[372] + local$$11217] === undefined) {
          this[_0x34b6[575]][local$$11216 + _0x34b6[372] + local$$11217] = new local$$11208(local$$11216, local$$11217);
        }
        return this[_0x34b6[575]][local$$11216 + _0x34b6[372] + local$$11217];
      };
      /** @type {function(): undefined} */
      local$$11192[_0x34b6[369]] = local$$11195;
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
        this[_0x34b6[554]] = null;
        this[_0x34b6[551]] = local$$11271;
        var local$$11286 = this;
        var local$$11290 = local$$11288(local$$11271);
        this[_0x34b6[553]] = (!local$$11272 ? this[_0x34b6[578]](local$$11273[_0x34b6[502]], local$$11290, local$$11273) : new Promise(function(local$$11303) {
          if (local$$11271[_0x34b6[442]][_0x34b6[441]][_0x34b6[579]] === _0x34b6[580] || local$$11271[_0x34b6[442]][_0x34b6[441]][_0x34b6[421]] == null) {
            /** @type {function(): undefined} */
            local$$11271[_0x34b6[442]][_0x34b6[443]] = local$$11271[_0x34b6[443]] = function() {
              local$$11303(local$$11271);
            };
          } else {
            local$$11303(local$$11271);
          }
        }))[_0x34b6[507]](function(local$$11358) {
          var local$$11363 = local$$11266(_0x34b6[577]);
          return local$$11363(local$$11358[_0x34b6[442]][_0x34b6[441]][_0x34b6[421]], {
            type : _0x34b6[446],
            width : local$$11358[_0x34b6[208]],
            height : local$$11358[_0x34b6[209]],
            proxy : local$$11273[_0x34b6[502]],
            javascriptEnabled : local$$11273[_0x34b6[422]],
            removeContainer : local$$11273[_0x34b6[498]],
            allowTaint : local$$11273[_0x34b6[497]],
            imageTimeout : local$$11273[_0x34b6[499]] / 2
          });
        })[_0x34b6[507]](function(local$$11408) {
          return local$$11286[_0x34b6[554]] = local$$11408;
        });
      }
      var local$$11425 = local$$11266(_0x34b6[487]);
      var local$$11288 = local$$11425[_0x34b6[491]];
      var local$$11437 = local$$11266(_0x34b6[490])[_0x34b6[489]];
      /**
       * @param {?} local$$11445
       * @param {?} local$$11446
       * @param {?} local$$11447
       * @return {?}
       */
      local$$11270[_0x34b6[219]][_0x34b6[578]] = function(local$$11445, local$$11446, local$$11447) {
        var local$$11452 = this[_0x34b6[551]];
        return local$$11437(local$$11452[_0x34b6[551]], local$$11445, local$$11452[_0x34b6[511]], local$$11446[_0x34b6[208]], local$$11446[_0x34b6[209]], local$$11447);
      };
      /** @type {function(?, ?, ?): undefined} */
      local$$11267[_0x34b6[369]] = local$$11270;
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
        this[_0x34b6[551]] = local$$11489[_0x34b6[275]];
        /** @type {!Array} */
        this[_0x34b6[581]] = [];
        /** @type {null} */
        this[_0x34b6[445]] = null;
        /** @type {number} */
        this[_0x34b6[582]] = .5;
        /** @type {number} */
        this[_0x34b6[583]] = .5;
        /** @type {number} */
        this[_0x34b6[584]] = .5;
        /** @type {number} */
        this[_0x34b6[585]] = .5;
        this[_0x34b6[553]] = Promise[_0x34b6[526]](true);
      }
      local$$11488[_0x34b6[586]] = {
        LINEAR : 1,
        RADIAL : 2
      };
      /** @type {!RegExp} */
      local$$11488[_0x34b6[587]] = /^\s*(rgba?\(\s*\d{1,3},\s*\d{1,3},\s*\d{1,3}(?:,\s*[0-9\.]+)?\s*\)|[a-z]{3,20}|#[a-f0-9]{3,6})(?:\s+(\d{1,3}(?:\.\d+)?)(%|px)?)?(?:\s|$)/i;
      /** @type {function(?): undefined} */
      local$$11485[_0x34b6[369]] = local$$11488;
    }, {}],
    10 : [function(local$$11572, local$$11573, local$$11574) {
      /**
       * @param {?} local$$11577
       * @param {?} local$$11578
       * @return {undefined}
       */
      function local$$11576(local$$11577, local$$11578) {
        this[_0x34b6[551]] = local$$11577;
        /** @type {!Image} */
        this[_0x34b6[554]] = new Image;
        var local$$11591 = this;
        /** @type {null} */
        this[_0x34b6[588]] = null;
        /** @type {!Promise} */
        this[_0x34b6[553]] = new Promise(function(local$$11602, local$$11603) {
          local$$11591[_0x34b6[554]][_0x34b6[443]] = local$$11602;
          local$$11591[_0x34b6[554]][_0x34b6[556]] = local$$11603;
          if (local$$11578) {
            local$$11591[_0x34b6[554]][_0x34b6[589]] = _0x34b6[590];
          }
          local$$11591[_0x34b6[554]][_0x34b6[551]] = local$$11577;
          if (local$$11591[_0x34b6[554]][_0x34b6[557]] === true) {
            local$$11602(local$$11591[_0x34b6[554]]);
          }
        });
      }
      /** @type {function(?, ?): undefined} */
      local$$11573[_0x34b6[369]] = local$$11576;
    }, {}],
    11 : [function(local$$11674, local$$11675, local$$11676) {
      /**
       * @param {?} local$$11679
       * @param {?} local$$11680
       * @return {undefined}
       */
      function local$$11678(local$$11679, local$$11680) {
        /** @type {null} */
        this[_0x34b6[600]] = null;
        this[_0x34b6[494]] = local$$11679;
        this[_0x34b6[601]] = local$$11680;
        this[_0x34b6[602]] = this[_0x34b6[604]](window[_0x34b6[603]][_0x34b6[549]]);
      }
      var local$$11718 = local$$11674(_0x34b6[396]);
      var local$$11723 = local$$11674(_0x34b6[591]);
      var local$$11728 = local$$11674(_0x34b6[592]);
      var local$$11733 = local$$11674(_0x34b6[593]);
      var local$$11738 = local$$11674(_0x34b6[594]);
      var local$$11743 = local$$11674(_0x34b6[595]);
      var local$$11748 = local$$11674(_0x34b6[596]);
      var local$$11753 = local$$11674(_0x34b6[597]);
      var local$$11758 = local$$11674(_0x34b6[598]);
      var local$$11766 = local$$11674(_0x34b6[487])[_0x34b6[599]];
      /**
       * @param {?} local$$11774
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[605]] = function(local$$11774) {
        /** @type {!Array} */
        var local$$11777 = [];
        local$$11774[_0x34b6[615]](function(local$$11782, local$$11783) {
          switch(local$$11783[_0x34b6[609]][_0x34b6[410]]) {
            case _0x34b6[612]:
              return local$$11782[_0x34b6[611]]([{
                args : [local$$11783[_0x34b6[609]][_0x34b6[551]]],
                method : _0x34b6[610]
              }]);
            case _0x34b6[613]:
            case _0x34b6[614]:
              return local$$11782[_0x34b6[611]]([{
                args : [local$$11783[_0x34b6[609]]],
                method : local$$11783[_0x34b6[609]][_0x34b6[410]]
              }]);
          }
          return local$$11782;
        }, [])[_0x34b6[608]](this[_0x34b6[607]](local$$11777, this[_0x34b6[606]]), this);
        return local$$11777;
      };
      /**
       * @param {?} local$$11867
       * @param {?} local$$11868
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[616]] = function(local$$11867, local$$11868) {
        local$$11868[_0x34b6[619]]()[_0x34b6[618]](this[_0x34b6[617]])[_0x34b6[608]](this[_0x34b6[607]](local$$11867, this[_0x34b6[606]]), this);
        return local$$11867;
      };
      /**
       * @param {?} local$$11904
       * @param {?} local$$11905
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[607]] = function(local$$11904, local$$11905) {
        return function(local$$11907) {
          local$$11907[_0x34b6[622]][_0x34b6[608]](function(local$$11915) {
            if (!this[_0x34b6[620]](local$$11904, local$$11915)) {
              local$$11904[_0x34b6[222]](0, 0, local$$11905[_0x34b6[238]](this, local$$11907));
              local$$11718(_0x34b6[621] + local$$11904[_0x34b6[223]], typeof local$$11915 === _0x34b6[501] ? local$$11915[_0x34b6[474]](0, 100) : local$$11915);
            }
          }, this);
        };
      };
      /**
       * @param {?} local$$11971
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[617]] = function(local$$11971) {
        return local$$11971[_0x34b6[623]] !== _0x34b6[624];
      };
      /**
       * @param {?} local$$11990
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[606]] = function(local$$11990) {
        if (local$$11990[_0x34b6[623]] === _0x34b6[610]) {
          var local$$12003 = local$$11990[_0x34b6[622]][0];
          if (this[_0x34b6[625]](local$$12003) && !this[_0x34b6[601]][_0x34b6[613]] && !this[_0x34b6[494]][_0x34b6[497]]) {
            return new local$$11743(local$$12003);
          } else {
            if (local$$12003[_0x34b6[473]](/data:image\/.*;base64,/i)) {
              return new local$$11723(local$$12003[_0x34b6[626]](/url\(['"]{0,}|['"]{0,}\)$/ig, _0x34b6[381]), false);
            } else {
              if (this[_0x34b6[627]](local$$12003) || this[_0x34b6[494]][_0x34b6[497]] === true || this[_0x34b6[625]](local$$12003)) {
                return new local$$11723(local$$12003, false);
              } else {
                if (this[_0x34b6[601]][_0x34b6[628]] && !this[_0x34b6[494]][_0x34b6[497]] && this[_0x34b6[494]][_0x34b6[629]]) {
                  return new local$$11723(local$$12003, true);
                } else {
                  if (this[_0x34b6[494]][_0x34b6[502]]) {
                    return new local$$11733(local$$12003, this[_0x34b6[494]][_0x34b6[502]]);
                  } else {
                    return new local$$11728(local$$12003);
                  }
                }
              }
            }
          }
        } else {
          if (local$$11990[_0x34b6[623]] === _0x34b6[630]) {
            return new local$$11753(local$$11990);
          } else {
            if (local$$11990[_0x34b6[623]] === _0x34b6[631]) {
              return new local$$11758(local$$11990);
            } else {
              if (local$$11990[_0x34b6[623]] === _0x34b6[613]) {
                return new local$$11748(local$$11990[_0x34b6[622]][0], this[_0x34b6[601]][_0x34b6[613]]);
              } else {
                if (local$$11990[_0x34b6[623]] === _0x34b6[614]) {
                  return new local$$11738(local$$11990[_0x34b6[622]][0], this[_0x34b6[627]](local$$11990[_0x34b6[622]][0][_0x34b6[551]]), this[_0x34b6[494]]);
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
      local$$11678[_0x34b6[219]][_0x34b6[625]] = function(local$$12211) {
        return local$$12211[_0x34b6[474]](local$$12211[_0x34b6[223]] - 3)[_0x34b6[387]]() === _0x34b6[613] || local$$11743[_0x34b6[219]][_0x34b6[632]](local$$12211);
      };
      /**
       * @param {?} local$$12248
       * @param {?} local$$12249
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[620]] = function(local$$12248, local$$12249) {
        return local$$12248[_0x34b6[633]](function(local$$12254) {
          return local$$12254[_0x34b6[551]] === local$$12249;
        });
      };
      /**
       * @param {?} local$$12275
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[627]] = function(local$$12275) {
        return this[_0x34b6[604]](local$$12275) === this[_0x34b6[602]];
      };
      /**
       * @param {?} local$$12296
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[604]] = function(local$$12296) {
        var local$$12312 = this[_0x34b6[600]] || (this[_0x34b6[600]] = document[_0x34b6[424]](_0x34b6[461]));
        local$$12312[_0x34b6[549]] = local$$12296;
        local$$12312[_0x34b6[549]] = local$$12312[_0x34b6[549]];
        return local$$12312[_0x34b6[634]] + local$$12312[_0x34b6[635]] + local$$12312[_0x34b6[636]];
      };
      /**
       * @param {?} local$$12349
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[637]] = function(local$$12349) {
        return this[_0x34b6[639]](local$$12349, this[_0x34b6[494]][_0x34b6[499]])[_0x34b6[638]](function() {
          var local$$12369 = new local$$11728(local$$12349[_0x34b6[551]]);
          return local$$12369[_0x34b6[553]][_0x34b6[507]](function(local$$12377) {
            local$$12349[_0x34b6[554]] = local$$12377;
          });
        });
      };
      /**
       * @param {?} local$$12402
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[640]] = function(local$$12402) {
        /** @type {null} */
        var local$$12405 = null;
        return this[_0x34b6[641]][_0x34b6[633]](function(local$$12413) {
          return (local$$12405 = local$$12413)[_0x34b6[551]] === local$$12402;
        }) ? local$$12405 : null;
      };
      /**
       * @param {?} local$$12437
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[642]] = function(local$$12437) {
        this[_0x34b6[641]] = local$$12437[_0x34b6[615]](local$$11766(this[_0x34b6[616]], this), this[_0x34b6[605]](local$$12437));
        this[_0x34b6[641]][_0x34b6[608]](function(local$$12462, local$$12463) {
          local$$12462[_0x34b6[553]][_0x34b6[507]](function() {
            local$$11718(_0x34b6[643] + (local$$12463 + 1), local$$12462);
          }, function(local$$12481) {
            local$$11718(_0x34b6[644] + (local$$12463 + 1), local$$12462, local$$12481);
          });
        });
        this[_0x34b6[528]] = Promise[_0x34b6[646]](this[_0x34b6[641]][_0x34b6[645]](this[_0x34b6[637]], this));
        local$$11718(_0x34b6[647]);
        return this;
      };
      /**
       * @param {?} local$$12532
       * @param {?} local$$12533
       * @return {?}
       */
      local$$11678[_0x34b6[219]][_0x34b6[639]] = function(local$$12532, local$$12533) {
        var local$$12535;
        var local$$12576 = Promise[_0x34b6[649]]([local$$12532[_0x34b6[553]], new Promise(function(local$$12543, local$$12544) {
          /** @type {number} */
          local$$12535 = setTimeout(function() {
            local$$11718(_0x34b6[648], local$$12532);
            local$$12544(local$$12532);
          }, local$$12533);
        })])[_0x34b6[507]](function(local$$12567) {
          clearTimeout(local$$12535);
          return local$$12567;
        });
        local$$12576[_0x34b6[638]](function() {
          clearTimeout(local$$12535);
        });
        return local$$12576;
      };
      /** @type {function(?, ?): undefined} */
      local$$11675[_0x34b6[369]] = local$$11678;
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
        local$$12620[_0x34b6[652]](this, arguments);
        this[_0x34b6[445]] = local$$12620[_0x34b6[586]][_0x34b6[653]];
        var local$$12664 = local$$12617[_0x34b6[654]][_0x34b6[386]](local$$12618[_0x34b6[622]][0]) || !local$$12620[_0x34b6[587]][_0x34b6[386]](local$$12618[_0x34b6[622]][0]);
        if (local$$12664) {
          local$$12618[_0x34b6[622]][0][_0x34b6[379]](/\s+/)[_0x34b6[659]]()[_0x34b6[608]](function(local$$12684, local$$12685) {
            switch(local$$12684) {
              case _0x34b6[432]:
                /** @type {number} */
                this[_0x34b6[582]] = 0;
                /** @type {number} */
                this[_0x34b6[584]] = 1;
                break;
              case _0x34b6[434]:
                /** @type {number} */
                this[_0x34b6[583]] = 0;
                /** @type {number} */
                this[_0x34b6[585]] = 1;
                break;
              case _0x34b6[655]:
                /** @type {number} */
                this[_0x34b6[582]] = 1;
                /** @type {number} */
                this[_0x34b6[584]] = 0;
                break;
              case _0x34b6[656]:
                /** @type {number} */
                this[_0x34b6[583]] = 1;
                /** @type {number} */
                this[_0x34b6[585]] = 0;
                break;
              case _0x34b6[657]:
                var local$$12760 = this[_0x34b6[583]];
                var local$$12765 = this[_0x34b6[582]];
                this[_0x34b6[583]] = this[_0x34b6[585]];
                this[_0x34b6[582]] = this[_0x34b6[584]];
                this[_0x34b6[584]] = local$$12765;
                this[_0x34b6[585]] = local$$12760;
                break;
              case _0x34b6[658]:
                break;
              default:
                /** @type {number} */
                var local$$12806 = parseFloat(local$$12684, 10) * .01;
                if (isNaN(local$$12806)) {
                  break;
                }
                if (local$$12685 === 0) {
                  /** @type {number} */
                  this[_0x34b6[583]] = local$$12806;
                  /** @type {number} */
                  this[_0x34b6[585]] = 1 - this[_0x34b6[583]];
                } else {
                  /** @type {number} */
                  this[_0x34b6[582]] = local$$12806;
                  /** @type {number} */
                  this[_0x34b6[584]] = 1 - this[_0x34b6[582]];
                }
                break;
            }
          }, this);
        } else {
          /** @type {number} */
          this[_0x34b6[583]] = 0;
          /** @type {number} */
          this[_0x34b6[585]] = 1;
        }
        this[_0x34b6[581]] = local$$12618[_0x34b6[622]][_0x34b6[388]](local$$12664 ? 1 : 0)[_0x34b6[645]](function(local$$12890) {
          var local$$12897 = local$$12890[_0x34b6[473]](local$$12620.REGEXP_COLORSTOP);
          /** @type {number} */
          var local$$12902 = +local$$12897[2];
          var local$$12911 = local$$12902 === 0 ? _0x34b6[660] : local$$12897[3];
          return {
            color : new local$$12913(local$$12897[1]),
            stop : local$$12911 === _0x34b6[660] ? local$$12902 / 100 : null
          };
        });
        if (this[_0x34b6[581]][0][_0x34b6[661]] === null) {
          /** @type {number} */
          this[_0x34b6[581]][0][_0x34b6[661]] = 0;
        }
        if (this[_0x34b6[581]][this[_0x34b6[581]][_0x34b6[223]] - 1][_0x34b6[661]] === null) {
          /** @type {number} */
          this[_0x34b6[581]][this[_0x34b6[581]][_0x34b6[223]] - 1][_0x34b6[661]] = 1;
        }
        this[_0x34b6[581]][_0x34b6[608]](function(local$$12999, local$$13000) {
          if (local$$12999[_0x34b6[661]] === null) {
            this[_0x34b6[581]][_0x34b6[388]](local$$13000)[_0x34b6[633]](function(local$$13017, local$$13018) {
              if (local$$13017[_0x34b6[661]] !== null) {
                local$$12999[_0x34b6[661]] = (local$$13017[_0x34b6[661]] - this[_0x34b6[581]][local$$13000 - 1][_0x34b6[661]]) / (local$$13018 + 1) + this[_0x34b6[581]][local$$13000 - 1][_0x34b6[661]];
                return true;
              } else {
                return false;
              }
            }, this);
          }
        }, this);
      }
      var local$$12620 = local$$12613(_0x34b6[650]);
      var local$$12913 = local$$12613(_0x34b6[651]);
      local$$12617[_0x34b6[219]] = Object[_0x34b6[242]](local$$12620[_0x34b6[219]]);
      /** @type {!RegExp} */
      local$$12617[_0x34b6[654]] = /^\s*(?:to|left|right|top|bottom|center|\d{1,3}(?:\.\d+)?%?)(?:\s|$)/i;
      /** @type {function(?): undefined} */
      local$$12614[_0x34b6[369]] = local$$12617;
    }, {
      "./color" : 3,
      "./gradientcontainer" : 9
    }],
    13 : [function(local$$13114, local$$13115, local$$13116) {
      /**
       * @return {undefined}
       */
      var local$$13119 = function() {
        if (local$$13119[_0x34b6[494]][_0x34b6[493]] && window[_0x34b6[662]] && window[_0x34b6[662]][_0x34b6[514]]) {
          Function[_0x34b6[219]][_0x34b6[599]][_0x34b6[238]](window[_0x34b6[662]][_0x34b6[514]], window[_0x34b6[662]])[_0x34b6[652]](window[_0x34b6[662]], [Date[_0x34b6[348]]() - local$$13119[_0x34b6[494]][_0x34b6[495]] + _0x34b6[663], _0x34b6[664]][_0x34b6[611]]([][_0x34b6[388]][_0x34b6[238]](arguments, 0)));
        }
      };
      local$$13119[_0x34b6[494]] = {
        logging : false
      };
      /** @type {function(): undefined} */
      local$$13115[_0x34b6[369]] = local$$13119;
    }, {}],
    14 : [function(local$$13217, local$$13218, local$$13219) {
      /**
       * @param {?} local$$13222
       * @param {?} local$$13223
       * @return {undefined}
       */
      function local$$13221(local$$13222, local$$13223) {
        this[_0x34b6[609]] = local$$13222;
        this[_0x34b6[667]] = local$$13223;
        /** @type {null} */
        this[_0x34b6[668]] = null;
        /** @type {null} */
        this[_0x34b6[669]] = null;
        /** @type {null} */
        this[_0x34b6[670]] = null;
        /** @type {!Array} */
        this[_0x34b6[671]] = [];
        /** @type {!Array} */
        this[_0x34b6[672]] = [];
        /** @type {null} */
        this[_0x34b6[666]] = null;
        /** @type {null} */
        this[_0x34b6[330]] = null;
        /** @type {null} */
        this[_0x34b6[673]] = null;
        this[_0x34b6[674]] = {};
        this[_0x34b6[675]] = {};
        /** @type {null} */
        this[_0x34b6[676]] = null;
        /** @type {null} */
        this[_0x34b6[677]] = null;
        /** @type {null} */
        this[_0x34b6[678]] = null;
        /** @type {boolean} */
        this[_0x34b6[679]] = false;
        /** @type {null} */
        this[_0x34b6[322]] = null;
      }
      /**
       * @param {?} local$$13328
       * @return {?}
       */
      function local$$13327(local$$13328) {
        var local$$13339 = local$$13328[_0x34b6[494]][local$$13328[_0x34b6[738]] || 0];
        return local$$13339 ? local$$13339[_0x34b6[739]] || _0x34b6[381] : _0x34b6[381];
      }
      /**
       * @param {!Array} local$$13354
       * @return {?}
       */
      function local$$13353(local$$13354) {
        if (local$$13354 && local$$13354[1] === _0x34b6[740]) {
          return local$$13354[2][_0x34b6[379]](_0x34b6[477])[_0x34b6[645]](function(local$$13373) {
            return parseFloat(local$$13373[_0x34b6[712]]());
          });
        } else {
          if (local$$13354 && local$$13354[1] === _0x34b6[741]) {
            var local$$13414 = local$$13354[2][_0x34b6[379]](_0x34b6[477])[_0x34b6[645]](function(local$$13403) {
              return parseFloat(local$$13403[_0x34b6[712]]());
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
        return local$$13437.toString()[_0x34b6[742]](_0x34b6[660]) !== -1;
      }
      /**
       * @param {?} local$$13453
       * @return {?}
       */
      function local$$13452(local$$13453) {
        return local$$13453[_0x34b6[626]](_0x34b6[450], _0x34b6[381]);
      }
      /**
       * @param {?} local$$13467
       * @return {?}
       */
      function local$$13466(local$$13467) {
        return parseFloat(local$$13467);
      }
      var local$$13476 = local$$13217(_0x34b6[651]);
      var local$$13481 = local$$13217(_0x34b6[487]);
      var local$$13486 = local$$13481[_0x34b6[491]];
      var local$$13491 = local$$13481[_0x34b6[665]];
      var local$$13496 = local$$13481[_0x34b6[666]];
      /**
       * @param {?} local$$13504
       * @return {undefined}
       */
      local$$13221[_0x34b6[219]][_0x34b6[680]] = function(local$$13504) {
        local$$13504[_0x34b6[330]] = this[_0x34b6[330]];
        local$$13504[_0x34b6[670]] = this[_0x34b6[670]];
        local$$13504[_0x34b6[669]] = this[_0x34b6[669]];
        local$$13504[_0x34b6[671]] = this[_0x34b6[671]];
        local$$13504[_0x34b6[672]] = this[_0x34b6[672]];
        local$$13504[_0x34b6[673]] = this[_0x34b6[673]];
        local$$13504[_0x34b6[675]] = this[_0x34b6[675]];
        local$$13504[_0x34b6[676]] = this[_0x34b6[676]];
        local$$13504[_0x34b6[322]] = this[_0x34b6[322]];
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[681]] = function() {
        return this[_0x34b6[322]] === null ? this[_0x34b6[322]] = this[_0x34b6[682]](_0x34b6[322]) : this[_0x34b6[322]];
      };
      /**
       * @param {?} local$$13619
       * @return {undefined}
       */
      local$$13221[_0x34b6[219]][_0x34b6[683]] = function(local$$13619) {
        this[_0x34b6[668]] = local$$13619;
        local$$13619[_0x34b6[684]][_0x34b6[220]](this);
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[685]] = function() {
        return this[_0x34b6[609]][_0x34b6[394]] === Node[_0x34b6[686]] ? this[_0x34b6[667]][_0x34b6[330]] : this[_0x34b6[688]](_0x34b6[687]) !== _0x34b6[624] && this[_0x34b6[688]](_0x34b6[427]) !== _0x34b6[429] && !this[_0x34b6[609]][_0x34b6[690]](_0x34b6[689]) && (this[_0x34b6[609]][_0x34b6[410]] !== _0x34b6[691] || this[_0x34b6[609]][_0x34b6[692]](_0x34b6[445]) !== _0x34b6[429]);
      };
      /**
       * @param {?} local$$13727
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[688]] = function(local$$13727) {
        if (!this[_0x34b6[673]]) {
          this[_0x34b6[673]] = this[_0x34b6[679]] ? this[_0x34b6[667]][_0x34b6[696]](this[_0x34b6[693]] ? _0x34b6[694] : _0x34b6[695]) : this[_0x34b6[696]](null);
        }
        return this[_0x34b6[675]][local$$13727] || (this[_0x34b6[675]][local$$13727] = this[_0x34b6[673]][local$$13727]);
      };
      /**
       * @param {?} local$$13790
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[697]] = function(local$$13790) {
        /** @type {!Array} */
        var local$$13801 = [_0x34b6[698], _0x34b6[699], _0x34b6[663], _0x34b6[700]];
        var local$$13807 = this[_0x34b6[688]](local$$13790);
        if (local$$13807 === undefined) {
          local$$13801[_0x34b6[633]](function(local$$13813) {
            local$$13807 = this[_0x34b6[688]](local$$13813 + local$$13790[_0x34b6[702]](0, 1)[_0x34b6[701]]() + local$$13790[_0x34b6[702]](1));
            return local$$13807 !== undefined;
          }, this);
        }
        return local$$13807 === undefined ? null : local$$13807;
      };
      /**
       * @param {?} local$$13861
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[696]] = function(local$$13861) {
        return this[_0x34b6[609]][_0x34b6[511]][_0x34b6[397]][_0x34b6[703]](this[_0x34b6[609]], local$$13861);
      };
      /**
       * @param {?} local$$13890
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[704]] = function(local$$13890) {
        /** @type {number} */
        var local$$13898 = parseInt(this[_0x34b6[688]](local$$13890), 10);
        return isNaN(local$$13898) ? 0 : local$$13898;
      };
      /**
       * @param {?} local$$13914
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[245]] = function(local$$13914) {
        return this[_0x34b6[674]][local$$13914] || (this[_0x34b6[674]][local$$13914] = new local$$13476(this[_0x34b6[688]](local$$13914)));
      };
      /**
       * @param {?} local$$13942
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[682]] = function(local$$13942) {
        /** @type {number} */
        var local$$13949 = parseFloat(this[_0x34b6[688]](local$$13942));
        return isNaN(local$$13949) ? 0 : local$$13949;
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[705]] = function() {
        var local$$13972 = this[_0x34b6[688]](_0x34b6[705]);
        switch(parseInt(local$$13972, 10)) {
          case 401:
            local$$13972 = _0x34b6[706];
            break;
          case 400:
            local$$13972 = _0x34b6[570];
            break;
        }
        return local$$13972;
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[707]] = function() {
        var local$$14017 = this[_0x34b6[688]](_0x34b6[671])[_0x34b6[473]](this.CLIP);
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
      local$$13221[_0x34b6[219]][_0x34b6[619]] = function() {
        return this[_0x34b6[676]] || (this[_0x34b6[676]] = local$$13491(this[_0x34b6[688]](_0x34b6[708])));
      };
      /**
       * @param {?} local$$14079
       * @param {number} local$$14080
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[709]] = function(local$$14079, local$$14080) {
        var local$$14095 = (this[_0x34b6[688]](local$$14079) || _0x34b6[381])[_0x34b6[379]](_0x34b6[477]);
        local$$14095 = local$$14095[local$$14080 || 0] || local$$14095[0] || _0x34b6[710];
        local$$14095 = local$$14095[_0x34b6[712]]()[_0x34b6[379]](_0x34b6[711]);
        if (local$$14095[_0x34b6[223]] === 1) {
          /** @type {!Array} */
          local$$14095 = [local$$14095[0], local$$13436(local$$14095[0]) ? _0x34b6[710] : local$$14095[0]];
        }
        return local$$14095;
      };
      /**
       * @param {?} local$$14152
       * @param {?} local$$14153
       * @param {?} local$$14154
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[713]] = function(local$$14152, local$$14153, local$$14154) {
        var local$$14162 = this[_0x34b6[709]](_0x34b6[714], local$$14154);
        var local$$14164;
        var local$$14166;
        if (local$$13436(local$$14162[0])) {
          /** @type {number} */
          local$$14164 = local$$14152[_0x34b6[208]] * parseFloat(local$$14162[0]) / 100;
        } else {
          if (/contain|cover/[_0x34b6[386]](local$$14162[0])) {
            /** @type {number} */
            var local$$14198 = local$$14152[_0x34b6[208]] / local$$14152[_0x34b6[209]];
            /** @type {number} */
            var local$$14207 = local$$14153[_0x34b6[208]] / local$$14153[_0x34b6[209]];
            return local$$14198 < local$$14207 ^ local$$14162[0] === _0x34b6[715] ? {
              width : local$$14152[_0x34b6[209]] * local$$14207,
              height : local$$14152[_0x34b6[209]]
            } : {
              width : local$$14152[_0x34b6[208]],
              height : local$$14152[_0x34b6[208]] / local$$14207
            };
          } else {
            /** @type {number} */
            local$$14164 = parseInt(local$$14162[0], 10);
          }
        }
        if (local$$14162[0] === _0x34b6[710] && local$$14162[1] === _0x34b6[710]) {
          local$$14166 = local$$14153[_0x34b6[209]];
        } else {
          if (local$$14162[1] === _0x34b6[710]) {
            /** @type {number} */
            local$$14166 = local$$14164 / local$$14153[_0x34b6[208]] * local$$14153[_0x34b6[209]];
          } else {
            if (local$$13436(local$$14162[1])) {
              /** @type {number} */
              local$$14166 = local$$14152[_0x34b6[209]] * parseFloat(local$$14162[1]) / 100;
            } else {
              /** @type {number} */
              local$$14166 = parseInt(local$$14162[1], 10);
            }
          }
        }
        if (local$$14162[0] === _0x34b6[710]) {
          /** @type {number} */
          local$$14164 = local$$14166 / local$$14153[_0x34b6[209]] * local$$14153[_0x34b6[208]];
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
      local$$13221[_0x34b6[219]][_0x34b6[716]] = function(local$$14337, local$$14338, local$$14339, local$$14340) {
        var local$$14348 = this[_0x34b6[709]](_0x34b6[717], local$$14339);
        var local$$14350;
        var local$$14352;
        if (local$$13436(local$$14348[0])) {
          /** @type {number} */
          local$$14350 = (local$$14337[_0x34b6[208]] - (local$$14340 || local$$14338)[_0x34b6[208]]) * (parseFloat(local$$14348[0]) / 100);
        } else {
          /** @type {number} */
          local$$14350 = parseInt(local$$14348[0], 10);
        }
        if (local$$14348[1] === _0x34b6[710]) {
          /** @type {number} */
          local$$14352 = local$$14350 / local$$14338[_0x34b6[208]] * local$$14338[_0x34b6[209]];
        } else {
          if (local$$13436(local$$14348[1])) {
            /** @type {number} */
            local$$14352 = (local$$14337[_0x34b6[209]] - (local$$14340 || local$$14338)[_0x34b6[209]]) * parseFloat(local$$14348[1]) / 100;
          } else {
            /** @type {number} */
            local$$14352 = parseInt(local$$14348[1], 10);
          }
        }
        if (local$$14348[0] === _0x34b6[710]) {
          /** @type {number} */
          local$$14350 = local$$14352 / local$$14338[_0x34b6[209]] * local$$14338[_0x34b6[208]];
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
      local$$13221[_0x34b6[219]][_0x34b6[718]] = function(local$$14460) {
        return this[_0x34b6[709]](_0x34b6[719], local$$14460)[0];
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[720]] = function() {
        var local$$14488 = this[_0x34b6[688]](_0x34b6[721]);
        /** @type {!Array} */
        var local$$14491 = [];
        if (local$$14488 && local$$14488 !== _0x34b6[624]) {
          var local$$14502 = local$$14488[_0x34b6[473]](this.TEXT_SHADOW_PROPERTY);
          /** @type {number} */
          var local$$14505 = 0;
          for (; local$$14502 && local$$14505 < local$$14502[_0x34b6[223]]; local$$14505++) {
            var local$$14520 = local$$14502[local$$14505][_0x34b6[473]](this.TEXT_SHADOW_VALUES);
            local$$14491[_0x34b6[220]]({
              color : new local$$13476(local$$14520[0]),
              offsetX : local$$14520[1] ? parseFloat(local$$14520[1][_0x34b6[626]](_0x34b6[450], _0x34b6[381])) : 0,
              offsetY : local$$14520[2] ? parseFloat(local$$14520[2][_0x34b6[626]](_0x34b6[450], _0x34b6[381])) : 0,
              blur : local$$14520[3] ? local$$14520[3][_0x34b6[626]](_0x34b6[450], _0x34b6[381]) : 0
            });
          }
        }
        return local$$14491;
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[722]] = function() {
        if (!this[_0x34b6[677]]) {
          if (this[_0x34b6[723]]()) {
            var local$$14604 = this[_0x34b6[724]]();
            var local$$14626 = this[_0x34b6[697]](_0x34b6[725])[_0x34b6[379]](_0x34b6[711])[_0x34b6[645]](local$$13452)[_0x34b6[645]](local$$13466);
            local$$14626[0] += local$$14604[_0x34b6[432]];
            local$$14626[1] += local$$14604[_0x34b6[434]];
            this[_0x34b6[677]] = {
              origin : local$$14626,
              matrix : this[_0x34b6[726]]()
            };
          } else {
            this[_0x34b6[677]] = {
              origin : [0, 0],
              matrix : [1, 0, 0, 1, 0, 0]
            };
          }
        }
        return this[_0x34b6[677]];
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[726]] = function() {
        if (!this[_0x34b6[678]]) {
          var local$$14699 = this[_0x34b6[697]](_0x34b6[727]);
          var local$$14709 = local$$14699 ? local$$13353(local$$14699[_0x34b6[473]](this.MATRIX_PROPERTY)) : null;
          this[_0x34b6[678]] = local$$14709 ? local$$14709 : [1, 0, 0, 1, 0, 0];
        }
        return this[_0x34b6[678]];
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[724]] = function() {
        return this[_0x34b6[669]] || (this[_0x34b6[669]] = this[_0x34b6[723]]() ? local$$13496(this[_0x34b6[609]]) : local$$13486(this[_0x34b6[609]]));
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[723]] = function() {
        return this[_0x34b6[726]]()[_0x34b6[2]](_0x34b6[477]) !== _0x34b6[728] || this[_0x34b6[667]] && this[_0x34b6[667]][_0x34b6[723]]();
      };
      /**
       * @return {?}
       */
      local$$13221[_0x34b6[219]][_0x34b6[729]] = function() {
        var local$$14821 = this[_0x34b6[609]][_0x34b6[275]] || _0x34b6[381];
        if (this[_0x34b6[609]][_0x34b6[730]] === _0x34b6[420]) {
          local$$14821 = local$$13327(this[_0x34b6[609]]);
        } else {
          if (this[_0x34b6[609]][_0x34b6[445]] === _0x34b6[731]) {
            local$$14821 = Array(local$$14821[_0x34b6[223]] + 1)[_0x34b6[2]](_0x34b6[732]);
          }
        }
        return local$$14821[_0x34b6[223]] === 0 ? this[_0x34b6[609]][_0x34b6[733]] || _0x34b6[381] : local$$14821;
      };
      /** @type {!RegExp} */
      local$$13221[_0x34b6[219]][_0x34b6[734]] = /(matrix|matrix3d)\((.+)\)/;
      /** @type {!RegExp} */
      local$$13221[_0x34b6[219]][_0x34b6[735]] = /((rgba|rgb)\([^\)]+\)(\s-?\d+px){0,})/g;
      /** @type {!RegExp} */
      local$$13221[_0x34b6[219]][_0x34b6[736]] = /(-?\d+px)|(#.+)|(rgb\(.+\))|(rgba\(.+\))/g;
      /** @type {!RegExp} */
      local$$13221[_0x34b6[219]][_0x34b6[737]] = /^rect\((\d+)px,? (\d+)px,? (\d+)px,? (\d+)px\)$/;
      /** @type {function(?, ?): undefined} */
      local$$13218[_0x34b6[369]] = local$$13221;
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
        local$$14950(_0x34b6[747]);
        this[_0x34b6[204]] = local$$14945;
        this[_0x34b6[494]] = local$$14948;
        /** @type {null} */
        this[_0x34b6[748]] = null;
        this[_0x34b6[601]] = local$$14946;
        /** @type {!Array} */
        this[_0x34b6[749]] = [];
        this[_0x34b6[668]] = new local$$14985(true, 1, local$$14944[_0x34b6[511]], null);
        var local$$14998 = new local$$14995(local$$14944, null);
        if (local$$14948[_0x34b6[343]]) {
          local$$14945[_0x34b6[750]](0, 0, local$$14945[_0x34b6[208]], local$$14945[_0x34b6[209]], new local$$15014(local$$14948[_0x34b6[343]]));
        }
        if (local$$14944 === local$$14944[_0x34b6[511]][_0x34b6[421]]) {
          var local$$15056 = new local$$14995(local$$14998[_0x34b6[245]](_0x34b6[751])[_0x34b6[469]]() ? local$$14944[_0x34b6[511]][_0x34b6[440]] : local$$14944[_0x34b6[511]][_0x34b6[421]], null);
          local$$14945[_0x34b6[750]](0, 0, local$$14945[_0x34b6[208]], local$$14945[_0x34b6[209]], local$$15056[_0x34b6[245]](_0x34b6[751]));
        }
        local$$14998[_0x34b6[752]] = local$$14998[_0x34b6[685]]();
        this[_0x34b6[753]](local$$14944[_0x34b6[511]]);
        this[_0x34b6[754]](local$$14944[_0x34b6[511]]);
        this[_0x34b6[755]] = local$$15108([local$$14998][_0x34b6[611]](this[_0x34b6[757]](local$$14998))[_0x34b6[618]](function(local$$15121) {
          return local$$15121[_0x34b6[330]] = local$$15121[_0x34b6[685]]();
        })[_0x34b6[645]](this[_0x34b6[756]], this));
        this[_0x34b6[758]] = new local$$15148;
        local$$14950(_0x34b6[759], this[_0x34b6[755]][_0x34b6[223]]);
        local$$14950(_0x34b6[760]);
        this[_0x34b6[761]]();
        local$$14950(_0x34b6[762]);
        this[_0x34b6[641]] = local$$14947[_0x34b6[642]](this[_0x34b6[755]][_0x34b6[618]](local$$15187));
        this[_0x34b6[528]] = this[_0x34b6[641]][_0x34b6[528]][_0x34b6[507]](local$$15204(function() {
          local$$14950(_0x34b6[763]);
          local$$14950(_0x34b6[764]);
          this[_0x34b6[765]]();
          local$$14950(_0x34b6[766]);
          this[_0x34b6[767]](this[_0x34b6[668]]);
          this[_0x34b6[768]](this[_0x34b6[668]]);
          local$$14950(_0x34b6[769] + this[_0x34b6[749]][_0x34b6[223]] + _0x34b6[770]);
          return new Promise(local$$15204(function(local$$15253) {
            if (!local$$14948[_0x34b6[496]]) {
              this[_0x34b6[749]][_0x34b6[608]](this[_0x34b6[771]], this);
              local$$15253();
            } else {
              if (typeof local$$14948[_0x34b6[496]] === _0x34b6[391]) {
                local$$14948[_0x34b6[496]][_0x34b6[238]](this, this[_0x34b6[749]], local$$15253);
              } else {
                if (this[_0x34b6[749]][_0x34b6[223]] > 0) {
                  /** @type {number} */
                  this[_0x34b6[772]] = 0;
                  this[_0x34b6[773]](this[_0x34b6[749]], local$$15253);
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
        return local$$15337[_0x34b6[667]] && local$$15337[_0x34b6[667]][_0x34b6[671]][_0x34b6[223]];
      }
      /**
       * @param {?} local$$15356
       * @return {?}
       */
      function local$$15355(local$$15356) {
        return local$$15356[_0x34b6[626]](/(\-[a-z])/g, function(local$$15363) {
          return local$$15363[_0x34b6[701]]()[_0x34b6[626]](_0x34b6[372], _0x34b6[381]);
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
        return local$$15389[_0x34b6[645]](function(local$$15397, local$$15398) {
          if (local$$15397[_0x34b6[208]] > 0) {
            var local$$15408 = local$$15390[_0x34b6[432]];
            var local$$15413 = local$$15390[_0x34b6[434]];
            var local$$15418 = local$$15390[_0x34b6[208]];
            /** @type {number} */
            var local$$15429 = local$$15390[_0x34b6[209]] - local$$15389[2][_0x34b6[208]];
            switch(local$$15398) {
              case 0:
                local$$15429 = local$$15389[0][_0x34b6[208]];
                local$$15397[_0x34b6[622]] = local$$15442({
                  c1 : [local$$15408, local$$15413],
                  c2 : [local$$15408 + local$$15418, local$$15413],
                  c3 : [local$$15408 + local$$15418 - local$$15389[1][_0x34b6[208]], local$$15413 + local$$15429],
                  c4 : [local$$15408 + local$$15389[3][_0x34b6[208]], local$$15413 + local$$15429]
                }, local$$15392[0], local$$15392[1], local$$15391[_0x34b6[879]], local$$15391[_0x34b6[880]], local$$15391[_0x34b6[881]], local$$15391[_0x34b6[882]]);
                break;
              case 1:
                /** @type {number} */
                local$$15408 = local$$15390[_0x34b6[432]] + local$$15390[_0x34b6[208]] - local$$15389[1][_0x34b6[208]];
                local$$15418 = local$$15389[1][_0x34b6[208]];
                local$$15397[_0x34b6[622]] = local$$15442({
                  c1 : [local$$15408 + local$$15418, local$$15413],
                  c2 : [local$$15408 + local$$15418, local$$15413 + local$$15429 + local$$15389[2][_0x34b6[208]]],
                  c3 : [local$$15408, local$$15413 + local$$15429],
                  c4 : [local$$15408, local$$15413 + local$$15389[0][_0x34b6[208]]]
                }, local$$15392[1], local$$15392[2], local$$15391[_0x34b6[881]], local$$15391[_0x34b6[882]], local$$15391[_0x34b6[883]], local$$15391[_0x34b6[884]]);
                break;
              case 2:
                /** @type {number} */
                local$$15413 = local$$15413 + local$$15390[_0x34b6[209]] - local$$15389[2][_0x34b6[208]];
                local$$15429 = local$$15389[2][_0x34b6[208]];
                local$$15397[_0x34b6[622]] = local$$15442({
                  c1 : [local$$15408 + local$$15418, local$$15413 + local$$15429],
                  c2 : [local$$15408, local$$15413 + local$$15429],
                  c3 : [local$$15408 + local$$15389[3][_0x34b6[208]], local$$15413],
                  c4 : [local$$15408 + local$$15418 - local$$15389[3][_0x34b6[208]], local$$15413]
                }, local$$15392[2], local$$15392[3], local$$15391[_0x34b6[883]], local$$15391[_0x34b6[884]], local$$15391[_0x34b6[885]], local$$15391[_0x34b6[886]]);
                break;
              case 3:
                local$$15418 = local$$15389[3][_0x34b6[208]];
                local$$15397[_0x34b6[622]] = local$$15442({
                  c1 : [local$$15408, local$$15413 + local$$15429 + local$$15389[2][_0x34b6[208]]],
                  c2 : [local$$15408, local$$15413],
                  c3 : [local$$15408 + local$$15418, local$$15413 + local$$15389[0][_0x34b6[208]]],
                  c4 : [local$$15408 + local$$15418, local$$15413 + local$$15429]
                }, local$$15392[3], local$$15392[0], local$$15391[_0x34b6[885]], local$$15391[_0x34b6[886]], local$$15391[_0x34b6[879]], local$$15391[_0x34b6[880]]);
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
        var local$$15703 = 4 * ((Math[_0x34b6[889]](2) - 1) / 3);
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
        var local$$15758 = local$$15751[_0x34b6[432]];
        var local$$15763 = local$$15751[_0x34b6[434]];
        var local$$15768 = local$$15751[_0x34b6[208]];
        var local$$15773 = local$$15751[_0x34b6[209]];
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
          topLeftOuter : local$$15686(local$$15758, local$$15763, local$$15789, local$$15805)[_0x34b6[891]][_0x34b6[890]](.5),
          topLeftInner : local$$15686(local$$15758 + local$$15753[3][_0x34b6[208]], local$$15763 + local$$15753[0][_0x34b6[208]], Math[_0x34b6[532]](0, local$$15789 - local$$15753[3][_0x34b6[208]]), Math[_0x34b6[532]](0, local$$15805 - local$$15753[0][_0x34b6[208]]))[_0x34b6[891]][_0x34b6[890]](.5),
          topRightOuter : local$$15686(local$$15758 + local$$15904, local$$15763, local$$15821, local$$15837)[_0x34b6[892]][_0x34b6[890]](.5),
          topRightInner : local$$15686(local$$15758 + Math[_0x34b6[472]](local$$15904, local$$15768 + local$$15753[3][_0x34b6[208]]), local$$15763 + local$$15753[0][_0x34b6[208]], local$$15904 > local$$15768 + local$$15753[3][_0x34b6[208]] ? 0 : local$$15821 - local$$15753[3][_0x34b6[208]], local$$15837 - local$$15753[0][_0x34b6[208]])[_0x34b6[892]][_0x34b6[890]](.5),
          bottomRightOuter : local$$15686(local$$15758 + local$$15910, local$$15763 + local$$15907, local$$15853, local$$15869)[_0x34b6[893]][_0x34b6[890]](.5),
          bottomRightInner : local$$15686(local$$15758 + Math[_0x34b6[472]](local$$15910, local$$15768 - local$$15753[3][_0x34b6[208]]), local$$15763 + Math[_0x34b6[472]](local$$15907, local$$15773 + local$$15753[0][_0x34b6[208]]), Math[_0x34b6[532]](0, local$$15853 - local$$15753[1][_0x34b6[208]]), local$$15869 - local$$15753[2][_0x34b6[208]])[_0x34b6[893]][_0x34b6[890]](.5),
          bottomLeftOuter : local$$15686(local$$15758, local$$15763 + local$$15913, local$$15885, local$$15901)[_0x34b6[894]][_0x34b6[890]](.5),
          bottomLeftInner : local$$15686(local$$15758 + local$$15753[3][_0x34b6[208]], local$$15763 + local$$15913, Math[_0x34b6[532]](0, local$$15885 - local$$15753[3][_0x34b6[208]]), local$$15901 - local$$15753[2][_0x34b6[208]])[_0x34b6[894]][_0x34b6[890]](.5)
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
            x : local$$16135[_0x34b6[290]] + (local$$16136[_0x34b6[290]] - local$$16135[_0x34b6[290]]) * local$$16137,
            y : local$$16135[_0x34b6[291]] + (local$$16136[_0x34b6[291]] - local$$16135[_0x34b6[291]]) * local$$16137
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
            local$$16197[_0x34b6[220]]([_0x34b6[895], local$$16131[_0x34b6[290]], local$$16131[_0x34b6[291]], local$$16132[_0x34b6[290]], local$$16132[_0x34b6[291]], local$$16133[_0x34b6[290]], local$$16133[_0x34b6[291]]]);
          },
          curveToReversed : function(local$$16227) {
            local$$16227[_0x34b6[220]]([_0x34b6[895], local$$16132[_0x34b6[290]], local$$16132[_0x34b6[291]], local$$16131[_0x34b6[290]], local$$16131[_0x34b6[291]], local$$16130[_0x34b6[290]], local$$16130[_0x34b6[291]]]);
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
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16264[1][_0x34b6[495]][_0x34b6[290]], local$$16264[1][_0x34b6[495]][_0x34b6[291]]]);
          local$$16264[1][_0x34b6[897]](local$$16270);
        } else {
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16261[_0x34b6[898]][0], local$$16261[_0x34b6[898]][1]]);
        }
        if (local$$16263[0] > 0 || local$$16263[1] > 0) {
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16266[0][_0x34b6[495]][_0x34b6[290]], local$$16266[0][_0x34b6[495]][_0x34b6[291]]]);
          local$$16266[0][_0x34b6[897]](local$$16270);
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16267[0][_0x34b6[899]][_0x34b6[290]], local$$16267[0][_0x34b6[899]][_0x34b6[291]]]);
          local$$16267[0][_0x34b6[900]](local$$16270);
        } else {
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16261[_0x34b6[901]][0], local$$16261[_0x34b6[901]][1]]);
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16261[_0x34b6[902]][0], local$$16261[_0x34b6[902]][1]]);
        }
        if (local$$16262[0] > 0 || local$$16262[1] > 0) {
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16265[1][_0x34b6[899]][_0x34b6[290]], local$$16265[1][_0x34b6[899]][_0x34b6[291]]]);
          local$$16265[1][_0x34b6[900]](local$$16270);
        } else {
          local$$16270[_0x34b6[220]]([_0x34b6[896], local$$16261[_0x34b6[903]][0], local$$16261[_0x34b6[903]][1]]);
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
          local$$16511[_0x34b6[220]]([_0x34b6[896], local$$16514[0][_0x34b6[495]][_0x34b6[290]], local$$16514[0][_0x34b6[495]][_0x34b6[291]]]);
          local$$16514[0][_0x34b6[897]](local$$16511);
          local$$16514[1][_0x34b6[897]](local$$16511);
        } else {
          local$$16511[_0x34b6[220]]([_0x34b6[896], local$$16516, local$$16517]);
        }
        if (local$$16513[0] > 0 || local$$16513[1] > 0) {
          local$$16511[_0x34b6[220]]([_0x34b6[896], local$$16515[0][_0x34b6[495]][_0x34b6[290]], local$$16515[0][_0x34b6[495]][_0x34b6[291]]]);
        }
      }
      /**
       * @param {?} local$$16616
       * @return {?}
       */
      function local$$16615(local$$16616) {
        return local$$16616[_0x34b6[704]](_0x34b6[904]) < 0;
      }
      /**
       * @param {?} local$$16630
       * @return {?}
       */
      function local$$16629(local$$16630) {
        return local$$16630[_0x34b6[704]](_0x34b6[904]) > 0;
      }
      /**
       * @param {?} local$$16644
       * @return {?}
       */
      function local$$16643(local$$16644) {
        return local$$16644[_0x34b6[704]](_0x34b6[904]) === 0;
      }
      /**
       * @param {?} local$$16658
       * @return {?}
       */
      function local$$16657(local$$16658) {
        return [_0x34b6[905], _0x34b6[906], _0x34b6[907]][_0x34b6[742]](local$$16658[_0x34b6[688]](_0x34b6[687])) !== -1;
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
        return local$$16690[_0x34b6[609]][_0x34b6[575]][_0x34b6[712]]()[_0x34b6[223]] > 0;
      }
      /**
       * @param {?} local$$16711
       * @return {?}
       */
      function local$$16710(local$$16711) {
        return /^(normal|none|0px)$/[_0x34b6[386]](local$$16711[_0x34b6[667]][_0x34b6[688]](_0x34b6[908]));
      }
      /**
       * @param {?} local$$16732
       * @return {?}
       */
      function local$$16731(local$$16732) {
        return [_0x34b6[910], _0x34b6[911], _0x34b6[912], _0x34b6[913]][_0x34b6[645]](function(local$$16746) {
          var local$$16758 = local$$16732[_0x34b6[688]](_0x34b6[436] + local$$16746 + _0x34b6[909]);
          var local$$16766 = local$$16758[_0x34b6[379]](_0x34b6[711]);
          if (local$$16766[_0x34b6[223]] <= 1) {
            local$$16766[1] = local$$16766[0];
          }
          return local$$16766[_0x34b6[645]](local$$16785);
        });
      }
      /**
       * @param {?} local$$16795
       * @return {?}
       */
      function local$$16794(local$$16795) {
        return local$$16795[_0x34b6[394]] === Node[_0x34b6[686]] || local$$16795[_0x34b6[394]] === Node[_0x34b6[786]];
      }
      /**
       * @param {?} local$$16816
       * @return {?}
       */
      function local$$16815(local$$16816) {
        var local$$16824 = local$$16816[_0x34b6[688]](_0x34b6[430]);
        var local$$16848 = [_0x34b6[451], _0x34b6[914], _0x34b6[431]][_0x34b6[742]](local$$16824) !== -1 ? local$$16816[_0x34b6[688]](_0x34b6[904]) : _0x34b6[710];
        return local$$16848 !== _0x34b6[710];
      }
      /**
       * @param {?} local$$16857
       * @return {?}
       */
      function local$$16856(local$$16857) {
        return local$$16857[_0x34b6[688]](_0x34b6[430]) !== _0x34b6[915];
      }
      /**
       * @param {?} local$$16872
       * @return {?}
       */
      function local$$16871(local$$16872) {
        return local$$16872[_0x34b6[688]](_0x34b6[916]) !== _0x34b6[624];
      }
      /**
       * @param {?} local$$16887
       * @return {?}
       */
      function local$$16886(local$$16887) {
        return [_0x34b6[906], _0x34b6[907]][_0x34b6[742]](local$$16887[_0x34b6[688]](_0x34b6[687])) !== -1;
      }
      /**
       * @param {!Function} local$$16910
       * @return {?}
       */
      function local$$16909(local$$16910) {
        var local$$16912 = this;
        return function() {
          return !local$$16910[_0x34b6[652]](local$$16912, arguments);
        };
      }
      /**
       * @param {?} local$$16926
       * @return {?}
       */
      function local$$15187(local$$16926) {
        return local$$16926[_0x34b6[609]][_0x34b6[394]] === Node[_0x34b6[786]];
      }
      /**
       * @param {?} local$$16942
       * @return {?}
       */
      function local$$16941(local$$16942) {
        return local$$16942[_0x34b6[679]] === true;
      }
      /**
       * @param {?} local$$16953
       * @return {?}
       */
      function local$$16952(local$$16953) {
        return local$$16953[_0x34b6[609]][_0x34b6[394]] === Node[_0x34b6[686]];
      }
      /**
       * @param {?} local$$16969
       * @return {?}
       */
      function local$$16968(local$$16969) {
        return function(local$$16971, local$$16972) {
          return local$$16971[_0x34b6[704]](_0x34b6[904]) + local$$16969[_0x34b6[742]](local$$16971) / local$$16969[_0x34b6[223]] - (local$$16972[_0x34b6[704]](_0x34b6[904]) + local$$16969[_0x34b6[742]](local$$16972) / local$$16969[_0x34b6[223]]);
        };
      }
      /**
       * @param {?} local$$17012
       * @return {?}
       */
      function local$$17011(local$$17012) {
        return local$$17012[_0x34b6[681]]() < 1;
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
        return local$$17031[_0x34b6[208]];
      }
      /**
       * @param {?} local$$17040
       * @return {?}
       */
      function local$$17039(local$$17040) {
        return local$$17040[_0x34b6[609]][_0x34b6[394]] !== Node[_0x34b6[786]] || [_0x34b6[411], _0x34b6[917], _0x34b6[918], _0x34b6[919], _0x34b6[920], _0x34b6[921]][_0x34b6[742]](local$$17040[_0x34b6[609]][_0x34b6[410]]) === -1;
      }
      /**
       * @param {!Array} local$$17081
       * @return {?}
       */
      function local$$15108(local$$17081) {
        return [][_0x34b6[611]][_0x34b6[652]]([], local$$17081);
      }
      /**
       * @param {?} local$$17096
       * @return {?}
       */
      function local$$17095(local$$17096) {
        var local$$17104 = local$$17096[_0x34b6[702]](0, 1);
        return local$$17104 === local$$17096[_0x34b6[702]](local$$17096[_0x34b6[223]] - 1) && local$$17104[_0x34b6[473]](/'|"/) ? local$$17096[_0x34b6[702]](1, local$$17096[_0x34b6[223]] - 2) : local$$17096;
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
        for (; local$$17138[_0x34b6[223]];) {
          if (local$$17156(local$$17138[local$$17144]) === local$$17147) {
            local$$17149 = local$$17138[_0x34b6[222]](0, local$$17144);
            if (local$$17149[_0x34b6[223]]) {
              local$$17141[_0x34b6[220]](local$$17173[_0x34b6[856]][_0x34b6[858]](local$$17149));
            }
            /** @type {boolean} */
            local$$17147 = !local$$17147;
            /** @type {number} */
            local$$17144 = 0;
          } else {
            local$$17144++;
          }
          if (local$$17144 >= local$$17138[_0x34b6[223]]) {
            local$$17149 = local$$17138[_0x34b6[222]](0, local$$17144);
            if (local$$17149[_0x34b6[223]]) {
              local$$17141[_0x34b6[220]](local$$17173[_0x34b6[856]][_0x34b6[858]](local$$17149));
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
        return [32, 13, 10, 9, 45][_0x34b6[742]](local$$17234) !== -1;
      }
      /**
       * @param {?} local$$17252
       * @return {?}
       */
      function local$$17251(local$$17252) {
        return /[^\u0000-\u00ff]/[_0x34b6[386]](local$$17252);
      }
      var local$$14950 = local$$14939(_0x34b6[396]);
      var local$$17173 = local$$14939(_0x34b6[393]);
      var local$$14995 = local$$14939(_0x34b6[486]);
      var local$$17278 = local$$14939(_0x34b6[743]);
      var local$$17283 = local$$14939(_0x34b6[744]);
      var local$$15148 = local$$14939(_0x34b6[745]);
      var local$$15014 = local$$14939(_0x34b6[651]);
      var local$$14985 = local$$14939(_0x34b6[746]);
      var local$$17300 = local$$14939(_0x34b6[487]);
      var local$$15204 = local$$17300[_0x34b6[599]];
      var local$$17309 = local$$17300[_0x34b6[491]];
      var local$$17314 = local$$17300[_0x34b6[665]];
      var local$$17319 = local$$17300[_0x34b6[666]];
      /**
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[761]] = function() {
        this[_0x34b6[755]][_0x34b6[608]](function(local$$17334) {
          if (local$$15187(local$$17334)) {
            if (local$$16941(local$$17334)) {
              local$$17334[_0x34b6[774]]();
            }
            local$$17334[_0x34b6[670]] = this[_0x34b6[775]](local$$17334);
            /** @type {!Array} */
            var local$$17373 = local$$17334[_0x34b6[688]](_0x34b6[385]) === _0x34b6[429] ? [local$$17334[_0x34b6[670]][_0x34b6[671]]] : [];
            var local$$17379 = local$$17334[_0x34b6[707]]();
            if (local$$17379 && [_0x34b6[451], _0x34b6[431]][_0x34b6[742]](local$$17334[_0x34b6[688]](_0x34b6[430])) !== -1) {
              local$$17373[_0x34b6[220]]([[_0x34b6[776], local$$17334[_0x34b6[669]][_0x34b6[432]] + local$$17379[_0x34b6[432]], local$$17334[_0x34b6[669]][_0x34b6[434]] + local$$17379[_0x34b6[434]], local$$17379[_0x34b6[655]] - local$$17379[_0x34b6[432]], local$$17379[_0x34b6[656]] - local$$17379[_0x34b6[434]]]]);
            }
            local$$17334[_0x34b6[671]] = local$$15336(local$$17334) ? local$$17334[_0x34b6[667]][_0x34b6[671]][_0x34b6[611]](local$$17373) : local$$17373;
            local$$17334[_0x34b6[672]] = local$$17334[_0x34b6[688]](_0x34b6[385]) !== _0x34b6[429] ? local$$17334[_0x34b6[671]][_0x34b6[611]]([local$$17334[_0x34b6[670]][_0x34b6[671]]]) : local$$17334[_0x34b6[671]];
            if (local$$16941(local$$17334)) {
              local$$17334[_0x34b6[777]]();
            }
          } else {
            if (local$$16952(local$$17334)) {
              local$$17334[_0x34b6[671]] = local$$15336(local$$17334) ? local$$17334[_0x34b6[667]][_0x34b6[671]] : [];
            }
          }
          if (!local$$16941(local$$17334)) {
            /** @type {null} */
            local$$17334[_0x34b6[669]] = null;
          }
        }, this);
      };
      /**
       * @param {!Object} local$$17547
       * @param {?} local$$17548
       * @param {number} local$$17549
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[773]] = function(local$$17547, local$$17548, local$$17549) {
        local$$17549 = local$$17549 || Date[_0x34b6[348]]();
        this[_0x34b6[771]](local$$17547[this[_0x34b6[772]]++]);
        if (local$$17547[_0x34b6[223]] === this[_0x34b6[772]]) {
          local$$17548();
        } else {
          if (local$$17549 + 20 > Date[_0x34b6[348]]()) {
            this[_0x34b6[773]](local$$17547, local$$17548, local$$17549);
          } else {
            setTimeout(local$$15204(function() {
              this[_0x34b6[773]](local$$17547, local$$17548);
            }, this), 0);
          }
        }
      };
      /**
       * @param {?} local$$17617
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[753]] = function(local$$17617) {
        this[_0x34b6[782]](local$$17617, _0x34b6[378] + local$$17283[_0x34b6[219]][_0x34b6[778]] + _0x34b6[779] + _0x34b6[378] + local$$17283[_0x34b6[219]][_0x34b6[780]] + _0x34b6[781]);
      };
      /**
       * @param {?} local$$17659
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[754]] = function(local$$17659) {
        this[_0x34b6[782]](local$$17659, _0x34b6[783] + _0x34b6[784]);
      };
      /**
       * @param {?} local$$17681
       * @param {?} local$$17682
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[782]] = function(local$$17681, local$$17682) {
        var local$$17690 = local$$17681[_0x34b6[424]](_0x34b6[428]);
        local$$17690[_0x34b6[785]] = local$$17682;
        local$$17681[_0x34b6[440]][_0x34b6[412]](local$$17690);
      };
      /**
       * @param {?} local$$17715
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[756]] = function(local$$17715) {
        /** @type {!Array} */
        var local$$17719 = [[local$$17715]];
        if (local$$17715[_0x34b6[609]][_0x34b6[394]] === Node[_0x34b6[786]]) {
          var local$$17737 = this[_0x34b6[787]](local$$17715, _0x34b6[694]);
          var local$$17745 = this[_0x34b6[787]](local$$17715, _0x34b6[695]);
          if (local$$17737) {
            local$$17719[_0x34b6[220]](local$$17737);
          }
          if (local$$17745) {
            local$$17719[_0x34b6[220]](local$$17745);
          }
        }
        return local$$15108(local$$17719);
      };
      /**
       * @param {(ArrayBuffer|ArrayBufferView|Blob|string)} local$$17777
       * @param {undefined} local$$17778
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[787]] = function(local$$17777, local$$17778) {
        var local$$17784 = local$$17777[_0x34b6[696]](local$$17778);
        if (!local$$17784 || !local$$17784[_0x34b6[788]] || local$$17784[_0x34b6[788]] === _0x34b6[624] || local$$17784[_0x34b6[788]] === _0x34b6[789] || local$$17784[_0x34b6[687]] === _0x34b6[624]) {
          return null;
        }
        var local$$17822 = local$$17095(local$$17784[_0x34b6[788]]);
        /** @type {boolean} */
        var local$$17833 = local$$17822[_0x34b6[702]](0, 3) === _0x34b6[610];
        var local$$17844 = document[_0x34b6[424]](local$$17833 ? _0x34b6[559] : _0x34b6[790]);
        var local$$17847 = new local$$17283(local$$17844, local$$17777, local$$17778);
        /** @type {number} */
        var local$$17854 = local$$17784[_0x34b6[223]] - 1;
        for (; local$$17854 >= 0; local$$17854--) {
          var local$$17865 = local$$15355(local$$17784[_0x34b6[791]](local$$17854));
          local$$17844[_0x34b6[428]][local$$17865] = local$$17784[local$$17865];
        }
        local$$17844[_0x34b6[425]] = local$$17283[_0x34b6[219]][_0x34b6[778]] + _0x34b6[711] + local$$17283[_0x34b6[219]][_0x34b6[780]];
        if (local$$17833) {
          local$$17844[_0x34b6[551]] = local$$17314(local$$17822)[0][_0x34b6[622]][0];
          return [local$$17847];
        } else {
          var local$$17918 = document[_0x34b6[407]](local$$17822);
          local$$17844[_0x34b6[412]](local$$17918);
          return [local$$17847, new local$$17278(local$$17918, local$$17847)];
        }
      };
      /**
       * @param {?} local$$17940
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[757]] = function(local$$17940) {
        return local$$15108([][_0x34b6[618]][_0x34b6[238]](local$$17940[_0x34b6[609]][_0x34b6[444]], local$$16794)[_0x34b6[645]](function(local$$17959) {
          var local$$17976 = [local$$17959[_0x34b6[394]] === Node[_0x34b6[686]] ? new local$$17278(local$$17959, local$$17940) : new local$$14995(local$$17959, local$$17940)][_0x34b6[618]](local$$17039);
          return local$$17959[_0x34b6[394]] === Node[_0x34b6[786]] && local$$17976[_0x34b6[223]] && local$$17959[_0x34b6[730]] !== _0x34b6[419] ? local$$17976[0][_0x34b6[685]]() ? local$$17976[_0x34b6[611]](this[_0x34b6[757]](local$$17976[0])) : [] : local$$17976;
        }, this));
      };
      /**
       * @param {?} local$$18031
       * @param {?} local$$18032
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[792]] = function(local$$18031, local$$18032) {
        var local$$18045 = new local$$14985(local$$18032, local$$18031[_0x34b6[681]](), local$$18031[_0x34b6[609]], local$$18031[_0x34b6[667]]);
        local$$18031[_0x34b6[680]](local$$18045);
        var local$$18063 = local$$18032 ? local$$18045[_0x34b6[793]](this) : local$$18045[_0x34b6[667]][_0x34b6[668]];
        local$$18063[_0x34b6[794]][_0x34b6[220]](local$$18045);
        local$$18031[_0x34b6[668]] = local$$18045;
      };
      /**
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[765]] = function() {
        this[_0x34b6[755]][_0x34b6[608]](function(local$$18095) {
          if (local$$15187(local$$18095) && (this[_0x34b6[795]](local$$18095) || local$$17011(local$$18095) || local$$16815(local$$18095) || this[_0x34b6[796]](local$$18095) || local$$18095[_0x34b6[723]]())) {
            this[_0x34b6[792]](local$$18095, true);
          } else {
            if (local$$15187(local$$18095) && (local$$16856(local$$18095) && local$$16643(local$$18095) || local$$16886(local$$18095) || local$$16871(local$$18095))) {
              this[_0x34b6[792]](local$$18095, false);
            } else {
              local$$18095[_0x34b6[683]](local$$18095[_0x34b6[667]][_0x34b6[668]]);
            }
          }
        }, this);
      };
      /**
       * @param {?} local$$18169
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[796]] = function(local$$18169) {
        return local$$18169[_0x34b6[609]][_0x34b6[410]] === _0x34b6[797] && local$$18169[_0x34b6[667]][_0x34b6[245]](_0x34b6[751])[_0x34b6[469]]();
      };
      /**
       * @param {?} local$$18205
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[795]] = function(local$$18205) {
        return local$$18205[_0x34b6[667]] === null;
      };
      /**
       * @param {?} local$$18223
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[767]] = function(local$$18223) {
        local$$18223[_0x34b6[794]][_0x34b6[798]](local$$16968(local$$18223[_0x34b6[794]][_0x34b6[388]](0)));
        local$$18223[_0x34b6[794]][_0x34b6[608]](this[_0x34b6[767]], this);
      };
      /**
       * @param {?} local$$18263
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[799]] = function(local$$18263) {
        return function(local$$18265, local$$18266, local$$18267) {
          if (local$$18263[_0x34b6[667]][_0x34b6[688]](_0x34b6[800])[_0x34b6[702]](0, 4) !== _0x34b6[624] || local$$18265[_0x34b6[712]]()[_0x34b6[223]] !== 0) {
            if (this[_0x34b6[601]][_0x34b6[801]] && !local$$18263[_0x34b6[667]][_0x34b6[723]]()) {
              var local$$18326 = local$$18267[_0x34b6[388]](0, local$$18266)[_0x34b6[2]](_0x34b6[381])[_0x34b6[223]];
              return this[_0x34b6[802]](local$$18263[_0x34b6[609]], local$$18326, local$$18265[_0x34b6[223]]);
            } else {
              if (local$$18263[_0x34b6[609]] && typeof local$$18263[_0x34b6[609]][_0x34b6[575]] === _0x34b6[501]) {
                var local$$18364 = local$$18263[_0x34b6[609]][_0x34b6[803]](local$$18265[_0x34b6[223]]);
                var local$$18380 = this[_0x34b6[804]](local$$18263[_0x34b6[609]], local$$18263[_0x34b6[667]][_0x34b6[723]]());
                local$$18263[_0x34b6[609]] = local$$18364;
                return local$$18380;
              }
            }
          } else {
            if (!this[_0x34b6[601]][_0x34b6[801]] || local$$18263[_0x34b6[667]][_0x34b6[723]]()) {
              local$$18263[_0x34b6[609]] = local$$18263[_0x34b6[609]][_0x34b6[803]](local$$18265[_0x34b6[223]]);
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
      local$$14943[_0x34b6[219]][_0x34b6[804]] = function(local$$18443, local$$18444) {
        var local$$18455 = local$$18443[_0x34b6[511]][_0x34b6[424]](_0x34b6[805]);
        var local$$18460 = local$$18443[_0x34b6[530]];
        var local$$18467 = local$$18443[_0x34b6[408]](true);
        local$$18455[_0x34b6[412]](local$$18443[_0x34b6[408]](true));
        local$$18460[_0x34b6[456]](local$$18455, local$$18443);
        var local$$18487 = local$$18444 ? local$$17319(local$$18455) : local$$17309(local$$18455);
        local$$18460[_0x34b6[456]](local$$18467, local$$18455);
        return local$$18487;
      };
      /**
       * @param {?} local$$18505
       * @param {(Object|number)} local$$18506
       * @param {!Object} local$$18507
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[802]] = function(local$$18505, local$$18506, local$$18507) {
        var local$$18524 = this[_0x34b6[748]] || (this[_0x34b6[748]] = local$$18505[_0x34b6[511]][_0x34b6[806]]());
        local$$18524[_0x34b6[807]](local$$18505, local$$18506);
        local$$18524[_0x34b6[808]](local$$18505, local$$18506 + local$$18507);
        return local$$18524[_0x34b6[809]]();
      };
      /**
       * @param {?} local$$18552
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[768]] = function(local$$18552) {
        var local$$18561 = local$$18552[_0x34b6[794]][_0x34b6[618]](local$$16615);
        var local$$18570 = local$$18552[_0x34b6[684]][_0x34b6[618]](local$$15187);
        var local$$18577 = local$$18570[_0x34b6[618]](local$$16909(local$$16871));
        var local$$18589 = local$$18577[_0x34b6[618]](local$$16909(local$$16856))[_0x34b6[618]](local$$16909(local$$16657));
        var local$$18600 = local$$18570[_0x34b6[618]](local$$16909(local$$16856))[_0x34b6[618]](local$$16871);
        var local$$18611 = local$$18577[_0x34b6[618]](local$$16909(local$$16856))[_0x34b6[618]](local$$16657);
        var local$$18628 = local$$18552[_0x34b6[794]][_0x34b6[611]](local$$18577[_0x34b6[618]](local$$16856))[_0x34b6[618]](local$$16643);
        var local$$18641 = local$$18552[_0x34b6[684]][_0x34b6[618]](local$$16952)[_0x34b6[618]](local$$16689);
        var local$$18650 = local$$18552[_0x34b6[794]][_0x34b6[618]](local$$16629);
        local$$18561[_0x34b6[611]](local$$18589)[_0x34b6[611]](local$$18600)[_0x34b6[611]](local$$18611)[_0x34b6[611]](local$$18628)[_0x34b6[611]](local$$18641)[_0x34b6[611]](local$$18650)[_0x34b6[608]](function(local$$18679) {
          this[_0x34b6[749]][_0x34b6[220]](local$$18679);
          if (local$$16682(local$$18679)) {
            this[_0x34b6[768]](local$$18679);
            this[_0x34b6[749]][_0x34b6[220]](new local$$15384);
          }
        }, this);
      };
      /**
       * @param {?} local$$18720
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[771]] = function(local$$18720) {
        try {
          if (local$$18720 instanceof local$$15384) {
            this[_0x34b6[204]][_0x34b6[811]][_0x34b6[810]]();
          } else {
            if (local$$16952(local$$18720)) {
              if (local$$16941(local$$18720[_0x34b6[667]])) {
                local$$18720[_0x34b6[667]][_0x34b6[774]]();
              }
              this[_0x34b6[812]](local$$18720);
              if (local$$16941(local$$18720[_0x34b6[667]])) {
                local$$18720[_0x34b6[667]][_0x34b6[777]]();
              }
            } else {
              this[_0x34b6[813]](local$$18720);
            }
          }
        } catch (local$$18781) {
          local$$14950(local$$18781);
          if (this[_0x34b6[494]][_0x34b6[500]]) {
            throw local$$18781;
          }
        }
      };
      /**
       * @param {?} local$$18807
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[813]] = function(local$$18807) {
        if (local$$16682(local$$18807)) {
          this[_0x34b6[204]][_0x34b6[814]](local$$18807[_0x34b6[322]]);
          this[_0x34b6[204]][_0x34b6[811]][_0x34b6[815]]();
          if (local$$18807[_0x34b6[723]]()) {
            this[_0x34b6[204]][_0x34b6[816]](local$$18807[_0x34b6[722]]());
          }
        }
        if (local$$18807[_0x34b6[609]][_0x34b6[410]] === _0x34b6[691] && local$$18807[_0x34b6[609]][_0x34b6[445]] === _0x34b6[817]) {
          this[_0x34b6[818]](local$$18807);
        } else {
          if (local$$18807[_0x34b6[609]][_0x34b6[410]] === _0x34b6[691] && local$$18807[_0x34b6[609]][_0x34b6[445]] === _0x34b6[819]) {
            this[_0x34b6[820]](local$$18807);
          } else {
            this[_0x34b6[821]](local$$18807);
          }
        }
      };
      /**
       * @param {?} local$$18922
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[821]] = function(local$$18922) {
        var local$$18928 = local$$18922[_0x34b6[724]]();
        this[_0x34b6[204]][_0x34b6[671]](local$$18922[_0x34b6[672]], function() {
          this[_0x34b6[204]][_0x34b6[822]](local$$18922, local$$18928, local$$18922[_0x34b6[670]][_0x34b6[670]][_0x34b6[645]](local$$17030));
        }, this);
        this[_0x34b6[204]][_0x34b6[671]](local$$18922[_0x34b6[671]], function() {
          this[_0x34b6[204]][_0x34b6[823]](local$$18922[_0x34b6[670]][_0x34b6[670]]);
        }, this);
        this[_0x34b6[204]][_0x34b6[671]](local$$18922[_0x34b6[672]], function() {
          switch(local$$18922[_0x34b6[609]][_0x34b6[410]]) {
            case _0x34b6[613]:
            case _0x34b6[614]:
              var local$$19023 = this[_0x34b6[641]][_0x34b6[640]](local$$18922[_0x34b6[609]]);
              if (local$$19023) {
                this[_0x34b6[204]][_0x34b6[824]](local$$18922, local$$18928, local$$18922[_0x34b6[670]], local$$19023);
              } else {
                local$$14950(_0x34b6[825] + local$$18922[_0x34b6[609]][_0x34b6[410]] + _0x34b6[826], local$$18922[_0x34b6[609]]);
              }
              break;
            case _0x34b6[612]:
              var local$$19075 = this[_0x34b6[641]][_0x34b6[640]](local$$18922[_0x34b6[609]][_0x34b6[551]]);
              if (local$$19075) {
                this[_0x34b6[204]][_0x34b6[824]](local$$18922, local$$18928, local$$18922[_0x34b6[670]], local$$19075);
              } else {
                local$$14950(_0x34b6[827], local$$18922[_0x34b6[609]][_0x34b6[551]]);
              }
              break;
            case _0x34b6[418]:
              this[_0x34b6[204]][_0x34b6[824]](local$$18922, local$$18928, local$$18922[_0x34b6[670]], {
                image : local$$18922[_0x34b6[609]]
              });
              break;
            case _0x34b6[420]:
            case _0x34b6[691]:
            case _0x34b6[419]:
              this[_0x34b6[828]](local$$18922);
              break;
          }
        }, this);
      };
      /**
       * @param {?} local$$19160
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[818]] = function(local$$19160) {
        var local$$19166 = local$$19160[_0x34b6[724]]();
        var local$$19178 = Math[_0x34b6[472]](local$$19166[_0x34b6[208]], local$$19166[_0x34b6[209]]);
        var local$$19191 = {
          width : local$$19178 - 1,
          height : local$$19178 - 1,
          top : local$$19166[_0x34b6[434]],
          left : local$$19166[_0x34b6[432]]
        };
        /** @type {!Array} */
        var local$$19196 = [3, 3];
        /** @type {!Array} */
        var local$$19199 = [local$$19196, local$$19196, local$$19196, local$$19196];
        var local$$19219 = [1, 1, 1, 1][_0x34b6[645]](function(local$$19209) {
          return {
            color : new local$$15014(_0x34b6[829]),
            width : local$$19209
          };
        });
        var local$$19222 = local$$15750(local$$19191, local$$19199, local$$19219);
        this[_0x34b6[204]][_0x34b6[671]](local$$19160[_0x34b6[672]], function() {
          this[_0x34b6[204]][_0x34b6[750]](local$$19191[_0x34b6[432]] + 1, local$$19191[_0x34b6[434]] + 1, local$$19191[_0x34b6[208]] - 2, local$$19191[_0x34b6[209]] - 2, new local$$15014(_0x34b6[830]));
          this[_0x34b6[204]][_0x34b6[823]](local$$15388(local$$19219, local$$19191, local$$19222, local$$19199));
          if (local$$19160[_0x34b6[609]][_0x34b6[831]]) {
            this[_0x34b6[204]][_0x34b6[834]](new local$$15014(_0x34b6[832]), _0x34b6[570], _0x34b6[570], _0x34b6[706], local$$19178 - 3 + _0x34b6[450], _0x34b6[833]);
            this[_0x34b6[204]][_0x34b6[739]](_0x34b6[835], local$$19191[_0x34b6[432]] + local$$19178 / 6, local$$19191[_0x34b6[434]] + local$$19178 - 1);
          }
        }, this);
      };
      /**
       * @param {?} local$$19342
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[820]] = function(local$$19342) {
        var local$$19348 = local$$19342[_0x34b6[724]]();
        /** @type {number} */
        var local$$19362 = Math[_0x34b6[472]](local$$19348[_0x34b6[208]], local$$19348[_0x34b6[209]]) - 2;
        this[_0x34b6[204]][_0x34b6[671]](local$$19342[_0x34b6[672]], function() {
          this[_0x34b6[204]][_0x34b6[836]](local$$19348[_0x34b6[432]] + 1, local$$19348[_0x34b6[434]] + 1, local$$19362, new local$$15014(_0x34b6[830]), 1, new local$$15014(_0x34b6[829]));
          if (local$$19342[_0x34b6[609]][_0x34b6[831]]) {
            this[_0x34b6[204]][_0x34b6[838]](Math[_0x34b6[837]](local$$19348[_0x34b6[432]] + local$$19362 / 4) + 1, Math[_0x34b6[837]](local$$19348[_0x34b6[434]] + local$$19362 / 4) + 1, Math[_0x34b6[376]](local$$19362 / 2), new local$$15014(_0x34b6[832]));
          }
        }, this);
      };
      /**
       * @param {?} local$$19462
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[828]] = function(local$$19462) {
        var local$$19468 = local$$19462[_0x34b6[729]]();
        if (local$$19468[_0x34b6[223]] > 0) {
          var local$$19481 = local$$19462[_0x34b6[609]][_0x34b6[511]];
          var local$$19489 = local$$19481[_0x34b6[424]](_0x34b6[805]);
          /** @type {!Array} */
          var local$$19530 = [_0x34b6[569], _0x34b6[839], _0x34b6[562], _0x34b6[705], _0x34b6[563], _0x34b6[245], _0x34b6[840], _0x34b6[841], _0x34b6[842], _0x34b6[843], _0x34b6[208], _0x34b6[209], _0x34b6[844], _0x34b6[845], _0x34b6[846], _0x34b6[847], _0x34b6[848], _0x34b6[849], _0x34b6[850]];
          local$$19530[_0x34b6[608]](function(local$$19535) {
            try {
              local$$19489[_0x34b6[428]][local$$19535] = local$$19462[_0x34b6[688]](local$$19535);
            } catch (local$$19548) {
              local$$14950(_0x34b6[851] + local$$19548[_0x34b6[852]]);
            }
          });
          var local$$19569 = local$$19462[_0x34b6[724]]();
          local$$19489[_0x34b6[428]][_0x34b6[430]] = _0x34b6[431];
          local$$19489[_0x34b6[428]][_0x34b6[432]] = local$$19569[_0x34b6[432]] + _0x34b6[450];
          local$$19489[_0x34b6[428]][_0x34b6[434]] = local$$19569[_0x34b6[434]] + _0x34b6[450];
          local$$19489[_0x34b6[853]] = local$$19468;
          local$$19481[_0x34b6[440]][_0x34b6[412]](local$$19489);
          this[_0x34b6[812]](new local$$17278(local$$19489[_0x34b6[409]], local$$19462));
          local$$19481[_0x34b6[440]][_0x34b6[529]](local$$19489);
        }
      };
      /**
       * @param {?} local$$19651
       * @return {undefined}
       */
      local$$14943[_0x34b6[219]][_0x34b6[812]] = function(local$$19651) {
        local$$19651[_0x34b6[854]]();
        var local$$19671 = local$$17173[_0x34b6[856]][_0x34b6[855]](local$$19651[_0x34b6[609]][_0x34b6[575]]);
        var local$$19710 = (!this[_0x34b6[494]][_0x34b6[857]] || local$$16710(local$$19651)) && !local$$17251(local$$19651[_0x34b6[609]][_0x34b6[575]]) ? local$$17137(local$$19671) : local$$19671[_0x34b6[645]](function(local$$19695) {
          return local$$17173[_0x34b6[856]][_0x34b6[858]]([local$$19695]);
        });
        var local$$19719 = local$$19651[_0x34b6[667]][_0x34b6[705]]();
        var local$$19730 = local$$19651[_0x34b6[667]][_0x34b6[688]](_0x34b6[563]);
        var local$$19741 = local$$19651[_0x34b6[667]][_0x34b6[688]](_0x34b6[562]);
        var local$$19750 = local$$19651[_0x34b6[667]][_0x34b6[720]]();
        this[_0x34b6[204]][_0x34b6[834]](local$$19651[_0x34b6[667]][_0x34b6[245]](_0x34b6[245]), local$$19651[_0x34b6[667]][_0x34b6[688]](_0x34b6[859]), local$$19651[_0x34b6[667]][_0x34b6[688]](_0x34b6[860]), local$$19719, local$$19730, local$$19741);
        if (local$$19750[_0x34b6[223]]) {
          this[_0x34b6[204]][_0x34b6[864]](local$$19750[0][_0x34b6[245]], local$$19750[0][_0x34b6[861]], local$$19750[0][_0x34b6[862]], local$$19750[0][_0x34b6[863]]);
        } else {
          this[_0x34b6[204]][_0x34b6[865]]();
        }
        this[_0x34b6[204]][_0x34b6[671]](local$$19651[_0x34b6[667]][_0x34b6[671]], function() {
          local$$19710[_0x34b6[645]](this[_0x34b6[799]](local$$19651), this)[_0x34b6[608]](function(local$$19854, local$$19855) {
            if (local$$19854) {
              this[_0x34b6[204]][_0x34b6[739]](local$$19710[local$$19855], local$$19854[_0x34b6[432]], local$$19854[_0x34b6[656]]);
              this[_0x34b6[866]](local$$19651[_0x34b6[667]], local$$19854, this[_0x34b6[758]][_0x34b6[576]](local$$19741, local$$19730));
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
      local$$14943[_0x34b6[219]][_0x34b6[866]] = function(local$$19907, local$$19908, local$$19909) {
        switch(local$$19907[_0x34b6[688]](_0x34b6[800])[_0x34b6[379]](_0x34b6[711])[0]) {
          case _0x34b6[867]:
            this[_0x34b6[204]][_0x34b6[750]](local$$19908[_0x34b6[432]], Math[_0x34b6[292]](local$$19908[_0x34b6[434]] + local$$19909[_0x34b6[567]] + local$$19909[_0x34b6[572]]), local$$19908[_0x34b6[208]], 1, local$$19907[_0x34b6[245]](_0x34b6[245]));
            break;
          case _0x34b6[868]:
            this[_0x34b6[204]][_0x34b6[750]](local$$19908[_0x34b6[432]], Math[_0x34b6[292]](local$$19908[_0x34b6[434]]), local$$19908[_0x34b6[208]], 1, local$$19907[_0x34b6[245]](_0x34b6[245]));
            break;
          case _0x34b6[869]:
            this[_0x34b6[204]][_0x34b6[750]](local$$19908[_0x34b6[432]], Math[_0x34b6[837]](local$$19908[_0x34b6[434]] + local$$19909[_0x34b6[573]] + local$$19909[_0x34b6[572]]), local$$19908[_0x34b6[208]], 1, local$$19907[_0x34b6[245]](_0x34b6[245]));
            break;
        }
      };
      var local$$20063 = {
        inset : [[_0x34b6[468], .6], [_0x34b6[468], .1], [_0x34b6[468], .1], [_0x34b6[468], .6]]
      };
      /**
       * @param {?} local$$20071
       * @return {?}
       */
      local$$14943[_0x34b6[219]][_0x34b6[775]] = function(local$$20071) {
        var local$$20077 = local$$20071[_0x34b6[724]]();
        var local$$20080 = local$$16731(local$$20071);
        var local$$20172 = [_0x34b6[874], _0x34b6[875], _0x34b6[876], _0x34b6[877]][_0x34b6[645]](function(local$$20094, local$$20095) {
          var local$$20107 = local$$20071[_0x34b6[688]](_0x34b6[436] + local$$20094 + _0x34b6[870]);
          var local$$20119 = local$$20071[_0x34b6[245]](_0x34b6[436] + local$$20094 + _0x34b6[871]);
          if (local$$20107 === _0x34b6[872] && local$$20119[_0x34b6[470]]()) {
            local$$20119 = new local$$15014([255, 255, 255, local$$20119[_0x34b6[461]]]);
          }
          var local$$20147 = local$$20063[local$$20107] ? local$$20063[local$$20107][local$$20095] : null;
          return {
            width : local$$20071[_0x34b6[704]](_0x34b6[436] + local$$20094 + _0x34b6[873]),
            color : local$$20147 ? local$$20119[local$$20147[0]](local$$20147[1]) : local$$20119,
            args : null
          };
        });
        var local$$20175 = local$$15750(local$$20077, local$$20080, local$$20172);
        return {
          clip : this[_0x34b6[878]](local$$20071, local$$20175, local$$20172, local$$20080, local$$20077),
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
      local$$14943[_0x34b6[219]][_0x34b6[878]] = function(local$$20194, local$$20195, local$$20196, local$$20197, local$$20198) {
        var local$$20206 = local$$20194[_0x34b6[688]](_0x34b6[672]);
        /** @type {!Array} */
        var local$$20209 = [];
        switch(local$$20206) {
          case _0x34b6[887]:
          case _0x34b6[888]:
            local$$16510(local$$20209, local$$20197[0], local$$20197[1], local$$20195[_0x34b6[880]], local$$20195[_0x34b6[882]], local$$20198[_0x34b6[432]] + local$$20196[3][_0x34b6[208]], local$$20198[_0x34b6[434]] + local$$20196[0][_0x34b6[208]]);
            local$$16510(local$$20209, local$$20197[1], local$$20197[2], local$$20195[_0x34b6[882]], local$$20195[_0x34b6[884]], local$$20198[_0x34b6[432]] + local$$20198[_0x34b6[208]] - local$$20196[1][_0x34b6[208]], local$$20198[_0x34b6[434]] + local$$20196[0][_0x34b6[208]]);
            local$$16510(local$$20209, local$$20197[2], local$$20197[3], local$$20195[_0x34b6[884]], local$$20195[_0x34b6[886]], local$$20198[_0x34b6[432]] + local$$20198[_0x34b6[208]] - local$$20196[1][_0x34b6[208]], local$$20198[_0x34b6[434]] + local$$20198[_0x34b6[209]] - local$$20196[2][_0x34b6[208]]);
            local$$16510(local$$20209, local$$20197[3], local$$20197[0], local$$20195[_0x34b6[886]], local$$20195[_0x34b6[880]], local$$20198[_0x34b6[432]] + local$$20196[3][_0x34b6[208]], local$$20198[_0x34b6[434]] + local$$20198[_0x34b6[209]] - local$$20196[2][_0x34b6[208]]);
            break;
          default:
            local$$16510(local$$20209, local$$20197[0], local$$20197[1], local$$20195[_0x34b6[879]], local$$20195[_0x34b6[881]], local$$20198[_0x34b6[432]], local$$20198[_0x34b6[434]]);
            local$$16510(local$$20209, local$$20197[1], local$$20197[2], local$$20195[_0x34b6[881]], local$$20195[_0x34b6[883]], local$$20198[_0x34b6[432]] + local$$20198[_0x34b6[208]], local$$20198[_0x34b6[434]]);
            local$$16510(local$$20209, local$$20197[2], local$$20197[3], local$$20195[_0x34b6[883]], local$$20195[_0x34b6[885]], local$$20198[_0x34b6[432]] + local$$20198[_0x34b6[208]], local$$20198[_0x34b6[434]] + local$$20198[_0x34b6[209]]);
            local$$16510(local$$20209, local$$20197[3], local$$20197[0], local$$20195[_0x34b6[885]], local$$20195[_0x34b6[879]], local$$20198[_0x34b6[432]], local$$20198[_0x34b6[434]] + local$$20198[_0x34b6[209]]);
            break;
        }
        return local$$20209;
      };
      /** @type {function(?, ?, ?, ?, ?): undefined} */
      local$$14940[_0x34b6[369]] = local$$14943;
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
        var local$$20488 = _0x34b6[924] in new XMLHttpRequest;
        if (!local$$20480) {
          return Promise[_0x34b6[504]](_0x34b6[925]);
        }
        var local$$20503 = local$$20501(local$$20488);
        var local$$20507 = local$$20505(local$$20480, local$$20479, local$$20503);
        return local$$20488 ? local$$20509(local$$20507) : local$$20511(local$$20481, local$$20507, local$$20503)[_0x34b6[507]](function(local$$20516) {
          return local$$20518(local$$20516[_0x34b6[788]]);
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
        var local$$20540 = _0x34b6[589] in new Image;
        var local$$20543 = local$$20501(local$$20540);
        var local$$20546 = local$$20505(local$$20533, local$$20532, local$$20543);
        return local$$20540 ? Promise[_0x34b6[526]](local$$20546) : local$$20511(local$$20534, local$$20546, local$$20543)[_0x34b6[507]](function(local$$20556) {
          return _0x34b6[926] + local$$20556[_0x34b6[445]] + _0x34b6[927] + local$$20556[_0x34b6[788]];
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
          var local$$20592 = local$$20579[_0x34b6[424]](_0x34b6[928]);
          /**
           * @return {undefined}
           */
          var local$$20614 = function() {
            delete window[_0x34b6[518]][_0x34b6[502]][local$$20581];
            local$$20579[_0x34b6[440]][_0x34b6[529]](local$$20592);
          };
          /**
           * @param {?} local$$20623
           * @return {undefined}
           */
          window[_0x34b6[518]][_0x34b6[502]][local$$20581] = function(local$$20623) {
            local$$20614();
            local$$20583(local$$20623);
          };
          local$$20592[_0x34b6[551]] = local$$20580;
          /**
           * @param {?} local$$20641
           * @return {undefined}
           */
          local$$20592[_0x34b6[556]] = function(local$$20641) {
            local$$20614();
            local$$20584(local$$20641);
          };
          local$$20579[_0x34b6[440]][_0x34b6[412]](local$$20592);
        });
      }
      /**
       * @param {boolean} local$$20665
       * @return {?}
       */
      function local$$20501(local$$20665) {
        return !local$$20665 ? _0x34b6[929] + Date[_0x34b6[348]]() + _0x34b6[930] + ++local$$20678 + _0x34b6[930] + Math[_0x34b6[292]](Math[_0x34b6[931]]() * 1E5) : _0x34b6[381];
      }
      /**
       * @param {?} local$$20701
       * @param {?} local$$20702
       * @param {?} local$$20703
       * @return {?}
       */
      function local$$20505(local$$20701, local$$20702, local$$20703) {
        return local$$20701 + _0x34b6[932] + encodeURIComponent(local$$20702) + (local$$20703[_0x34b6[223]] ? _0x34b6[933] + local$$20703 : _0x34b6[381]);
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
            local$$20733 = local$$20731[_0x34b6[935]](local$$20727, _0x34b6[934]);
          } catch (local$$20744) {
            local$$20745(_0x34b6[936]);
            local$$20733 = document[_0x34b6[938]][_0x34b6[937]](_0x34b6[381]);
            try {
              local$$20733[_0x34b6[452]]();
              local$$20733[_0x34b6[454]](local$$20727);
              local$$20733[_0x34b6[457]]();
            } catch (local$$20777) {
              local$$20745(_0x34b6[939]);
              local$$20733[_0x34b6[440]][_0x34b6[785]] = local$$20727;
            }
          }
          var local$$20805 = local$$20733[_0x34b6[524]](_0x34b6[940]);
          if (!local$$20805 || !local$$20805[_0x34b6[549]][_0x34b6[941]]) {
            var local$$20822 = local$$20733[_0x34b6[424]](_0x34b6[940]);
            local$$20822[_0x34b6[549]] = local$$20725;
            local$$20733[_0x34b6[942]][_0x34b6[943]](local$$20822, local$$20733[_0x34b6[942]][_0x34b6[409]]);
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
        return (new local$$20478(local$$20853, local$$20854, window[_0x34b6[441]]))[_0x34b6[507]](local$$20724(local$$20853))[_0x34b6[507]](function(local$$20872) {
          return local$$20874(local$$20872, local$$20855, local$$20856, local$$20857, local$$20858, 0, 0);
        });
      }
      var local$$20509 = local$$20474(_0x34b6[922]);
      var local$$20892 = local$$20474(_0x34b6[487]);
      var local$$20745 = local$$20474(_0x34b6[396]);
      var local$$20874 = local$$20474(_0x34b6[488]);
      var local$$20518 = local$$20892[_0x34b6[923]];
      /** @type {number} */
      var local$$20678 = 0;
      /** @type {function(?, ?, ?): ?} */
      local$$20476[_0x34b6[944]] = local$$20478;
      /** @type {function(?, ?, ?): ?} */
      local$$20476[_0x34b6[945]] = local$$20531;
      /** @type {function(?, boolean, ?, ?, ?, ?): ?} */
      local$$20476[_0x34b6[489]] = local$$20852;
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
        var local$$20945 = document[_0x34b6[424]](_0x34b6[461]);
        local$$20945[_0x34b6[549]] = local$$20936;
        local$$20936 = local$$20945[_0x34b6[549]];
        this[_0x34b6[551]] = local$$20936;
        /** @type {!Image} */
        this[_0x34b6[554]] = new Image;
        var local$$20968 = this;
        /** @type {!Promise} */
        this[_0x34b6[553]] = new Promise(function(local$$20973, local$$20974) {
          local$$20968[_0x34b6[554]][_0x34b6[589]] = _0x34b6[946];
          local$$20968[_0x34b6[554]][_0x34b6[443]] = local$$20973;
          local$$20968[_0x34b6[554]][_0x34b6[556]] = local$$20974;
          (new local$$21002(local$$20936, local$$20937, document))[_0x34b6[507]](function(local$$21007) {
            local$$20968[_0x34b6[554]][_0x34b6[551]] = local$$21007;
          })[_0x34b6[638]](local$$20974);
        });
      }
      var local$$21002 = local$$20931(_0x34b6[490])[_0x34b6[945]];
      /** @type {function(?, !Object): undefined} */
      local$$20932[_0x34b6[369]] = local$$20935;
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
        local$$21059[_0x34b6[238]](this, local$$21055, local$$21056);
        /** @type {boolean} */
        this[_0x34b6[679]] = true;
        /** @type {boolean} */
        this[_0x34b6[693]] = local$$21057 === _0x34b6[694];
      }
      var local$$21059 = local$$21050(_0x34b6[486]);
      /**
       * @param {?} local$$21091
       * @return {undefined}
       */
      local$$21054[_0x34b6[219]][_0x34b6[680]] = function(local$$21091) {
        local$$21054[_0x34b6[219]][_0x34b6[680]][_0x34b6[238]](this, local$$21091);
        /** @type {boolean} */
        local$$21091[_0x34b6[679]] = true;
        local$$21091[_0x34b6[693]] = this[_0x34b6[693]];
      };
      local$$21054[_0x34b6[219]] = Object[_0x34b6[242]](local$$21059[_0x34b6[219]]);
      /**
       * @return {undefined}
       */
      local$$21054[_0x34b6[219]][_0x34b6[774]] = function() {
        if (this[_0x34b6[693]]) {
          this[_0x34b6[667]][_0x34b6[609]][_0x34b6[943]](this[_0x34b6[609]], this[_0x34b6[667]][_0x34b6[609]][_0x34b6[409]]);
        } else {
          this[_0x34b6[667]][_0x34b6[609]][_0x34b6[412]](this[_0x34b6[609]]);
        }
        this[_0x34b6[667]][_0x34b6[609]][_0x34b6[425]] += _0x34b6[711] + this[_0x34b6[947]]();
      };
      /**
       * @return {undefined}
       */
      local$$21054[_0x34b6[219]][_0x34b6[777]] = function() {
        this[_0x34b6[609]][_0x34b6[530]][_0x34b6[529]](this[_0x34b6[609]]);
        this[_0x34b6[667]][_0x34b6[609]][_0x34b6[425]] = this[_0x34b6[667]][_0x34b6[609]][_0x34b6[425]][_0x34b6[626]](this[_0x34b6[947]](), _0x34b6[381]);
      };
      /**
       * @return {?}
       */
      local$$21054[_0x34b6[219]][_0x34b6[947]] = function() {
        return this[_0x34b6[948] + (this[_0x34b6[693]] ? _0x34b6[949] : _0x34b6[950])];
      };
      local$$21054[_0x34b6[219]][_0x34b6[778]] = _0x34b6[951];
      local$$21054[_0x34b6[219]][_0x34b6[780]] = _0x34b6[952];
      /** @type {function(?, ?, ?): undefined} */
      local$$21051[_0x34b6[369]] = local$$21054;
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
        this[_0x34b6[208]] = local$$21322;
        this[_0x34b6[209]] = local$$21323;
        this[_0x34b6[641]] = local$$21324;
        this[_0x34b6[494]] = local$$21325;
        this[_0x34b6[441]] = local$$21326;
      }
      var local$$21358 = local$$21317(_0x34b6[396]);
      /**
       * @param {?} local$$21366
       * @param {?} local$$21367
       * @param {?} local$$21368
       * @param {?} local$$21369
       * @return {undefined}
       */
      local$$21321[_0x34b6[219]][_0x34b6[824]] = function(local$$21366, local$$21367, local$$21368, local$$21369) {
        var local$$21377 = local$$21366[_0x34b6[704]](_0x34b6[840]);
        var local$$21385 = local$$21366[_0x34b6[704]](_0x34b6[841]);
        var local$$21393 = local$$21366[_0x34b6[704]](_0x34b6[842]);
        var local$$21401 = local$$21366[_0x34b6[704]](_0x34b6[843]);
        var local$$21406 = local$$21368[_0x34b6[670]];
        /** @type {number} */
        var local$$21425 = local$$21367[_0x34b6[208]] - (local$$21406[1][_0x34b6[208]] + local$$21406[3][_0x34b6[208]] + local$$21377 + local$$21393);
        /** @type {number} */
        var local$$21444 = local$$21367[_0x34b6[209]] - (local$$21406[0][_0x34b6[208]] + local$$21406[2][_0x34b6[208]] + local$$21385 + local$$21401);
        this[_0x34b6[542]](local$$21369, 0, 0, local$$21369[_0x34b6[554]][_0x34b6[208]] || local$$21425, local$$21369[_0x34b6[554]][_0x34b6[209]] || local$$21444, local$$21367[_0x34b6[432]] + local$$21377 + local$$21406[3][_0x34b6[208]], local$$21367[_0x34b6[434]] + local$$21385 + local$$21406[0][_0x34b6[208]], local$$21425, local$$21444);
      };
      /**
       * @param {?} local$$21497
       * @param {?} local$$21498
       * @param {?} local$$21499
       * @return {undefined}
       */
      local$$21321[_0x34b6[219]][_0x34b6[822]] = function(local$$21497, local$$21498, local$$21499) {
        if (local$$21498[_0x34b6[209]] > 0 && local$$21498[_0x34b6[208]] > 0) {
          this[_0x34b6[953]](local$$21497, local$$21498);
          this[_0x34b6[954]](local$$21497, local$$21498, local$$21499);
        }
      };
      /**
       * @param {?} local$$21534
       * @param {?} local$$21535
       * @return {undefined}
       */
      local$$21321[_0x34b6[219]][_0x34b6[953]] = function(local$$21534, local$$21535) {
        var local$$21543 = local$$21534[_0x34b6[245]](_0x34b6[751]);
        if (!local$$21543[_0x34b6[469]]()) {
          this[_0x34b6[750]](local$$21535[_0x34b6[432]], local$$21535[_0x34b6[434]], local$$21535[_0x34b6[208]], local$$21535[_0x34b6[209]], local$$21543);
        }
      };
      /**
       * @param {?} local$$21579
       * @return {undefined}
       */
      local$$21321[_0x34b6[219]][_0x34b6[823]] = function(local$$21579) {
        local$$21579[_0x34b6[608]](this[_0x34b6[955]], this);
      };
      /**
       * @param {?} local$$21599
       * @return {undefined}
       */
      local$$21321[_0x34b6[219]][_0x34b6[955]] = function(local$$21599) {
        if (!local$$21599[_0x34b6[245]][_0x34b6[469]]() && local$$21599[_0x34b6[622]] !== null) {
          this[_0x34b6[956]](local$$21599[_0x34b6[622]], local$$21599[_0x34b6[245]]);
        }
      };
      /**
       * @param {?} local$$21638
       * @param {?} local$$21639
       * @param {?} local$$21640
       * @return {undefined}
       */
      local$$21321[_0x34b6[219]][_0x34b6[954]] = function(local$$21638, local$$21639, local$$21640) {
        var local$$21646 = local$$21638[_0x34b6[619]]();
        local$$21646[_0x34b6[659]]()[_0x34b6[608]](function(local$$21655, local$$21656, local$$21657) {
          switch(local$$21655[_0x34b6[623]]) {
            case _0x34b6[610]:
              var local$$21676 = this[_0x34b6[641]][_0x34b6[640]](local$$21655[_0x34b6[622]][0]);
              if (local$$21676) {
                this[_0x34b6[957]](local$$21638, local$$21639, local$$21676, local$$21657[_0x34b6[223]] - (local$$21656 + 1), local$$21640);
              } else {
                local$$21358(_0x34b6[958], local$$21655[_0x34b6[622]][0]);
              }
              break;
            case _0x34b6[630]:
            case _0x34b6[631]:
              var local$$21722 = this[_0x34b6[641]][_0x34b6[640]](local$$21655[_0x34b6[275]]);
              if (local$$21722) {
                this[_0x34b6[959]](local$$21722, local$$21639, local$$21640);
              } else {
                local$$21358(_0x34b6[958], local$$21655[_0x34b6[622]][0]);
              }
              break;
            case _0x34b6[624]:
              break;
            default:
              local$$21358(_0x34b6[960], local$$21655[_0x34b6[622]][0]);
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
      local$$21321[_0x34b6[219]][_0x34b6[957]] = function(local$$21776, local$$21777, local$$21778, local$$21779, local$$21780) {
        var local$$21789 = local$$21776[_0x34b6[713]](local$$21777, local$$21778[_0x34b6[554]], local$$21779);
        var local$$21798 = local$$21776[_0x34b6[716]](local$$21777, local$$21778[_0x34b6[554]], local$$21779, local$$21789);
        var local$$21804 = local$$21776[_0x34b6[718]](local$$21779);
        switch(local$$21804) {
          case _0x34b6[961]:
          case _0x34b6[963]:
            this[_0x34b6[962]](local$$21778, local$$21798, local$$21789, local$$21777, local$$21777[_0x34b6[432]] + local$$21780[3], local$$21777[_0x34b6[434]] + local$$21798[_0x34b6[434]] + local$$21780[0], 99999, local$$21789[_0x34b6[209]], local$$21780);
            break;
          case _0x34b6[964]:
          case _0x34b6[965]:
            this[_0x34b6[962]](local$$21778, local$$21798, local$$21789, local$$21777, local$$21777[_0x34b6[432]] + local$$21798[_0x34b6[432]] + local$$21780[3], local$$21777[_0x34b6[434]] + local$$21780[0], local$$21789[_0x34b6[208]], 99999, local$$21780);
            break;
          case _0x34b6[966]:
            this[_0x34b6[962]](local$$21778, local$$21798, local$$21789, local$$21777, local$$21777[_0x34b6[432]] + local$$21798[_0x34b6[432]] + local$$21780[3], local$$21777[_0x34b6[434]] + local$$21798[_0x34b6[434]] + local$$21780[0], local$$21789[_0x34b6[208]], local$$21789[_0x34b6[209]], local$$21780);
            break;
          default:
            this[_0x34b6[967]](local$$21778, local$$21798, local$$21789, {
              top : local$$21777[_0x34b6[434]],
              left : local$$21777[_0x34b6[432]]
            }, local$$21780[3], local$$21780[0]);
            break;
        }
      };
      /** @type {function(?, ?, ?, ?, ?): undefined} */
      local$$21318[_0x34b6[369]] = local$$21321;
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
        local$$21955[_0x34b6[652]](this, arguments);
        this[_0x34b6[516]] = this[_0x34b6[494]][_0x34b6[516]] || this[_0x34b6[441]][_0x34b6[424]](_0x34b6[516]);
        if (!this[_0x34b6[494]][_0x34b6[516]]) {
          this[_0x34b6[516]][_0x34b6[208]] = local$$21952;
          this[_0x34b6[516]][_0x34b6[209]] = local$$21953;
        }
        this[_0x34b6[811]] = this[_0x34b6[516]][_0x34b6[403]](_0x34b6[402]);
        this[_0x34b6[971]] = this[_0x34b6[441]][_0x34b6[424]](_0x34b6[516])[_0x34b6[403]](_0x34b6[402]);
        this[_0x34b6[811]][_0x34b6[972]] = _0x34b6[656];
        this[_0x34b6[973]] = {};
        local$$22058(_0x34b6[974], local$$21952, _0x34b6[290], local$$21953);
      }
      /**
       * @param {?} local$$22068
       * @return {?}
       */
      function local$$22067(local$$22068) {
        return local$$22068[_0x34b6[223]] > 0;
      }
      var local$$21955 = local$$21947(_0x34b6[968]);
      var local$$22085 = local$$21947(_0x34b6[969]);
      var local$$22058 = local$$21947(_0x34b6[970]);
      local$$21951[_0x34b6[219]] = Object[_0x34b6[242]](local$$21955[_0x34b6[219]]);
      /**
       * @param {string} local$$22109
       * @return {?}
       */
      local$$21951[_0x34b6[219]][_0x34b6[975]] = function(local$$22109) {
        this[_0x34b6[811]][_0x34b6[976]] = typeof local$$22109 === _0x34b6[368] && !!local$$22109[_0x34b6[481]] ? local$$22109.toString() : local$$22109;
        return this[_0x34b6[811]];
      };
      /**
       * @param {?} local$$22146
       * @param {?} local$$22147
       * @param {?} local$$22148
       * @param {?} local$$22149
       * @param {?} local$$22150
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[750]] = function(local$$22146, local$$22147, local$$22148, local$$22149, local$$22150) {
        this[_0x34b6[975]](local$$22150)[_0x34b6[977]](local$$22146, local$$22147, local$$22148, local$$22149);
      };
      /**
       * @param {number} local$$22171
       * @param {number} local$$22172
       * @param {number} local$$22173
       * @param {?} local$$22174
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[838]] = function(local$$22171, local$$22172, local$$22173, local$$22174) {
        this[_0x34b6[975]](local$$22174);
        this[_0x34b6[811]][_0x34b6[978]]();
        this[_0x34b6[811]][_0x34b6[980]](local$$22171 + local$$22173 / 2, local$$22172 + local$$22173 / 2, local$$22173 / 2, 0, Math[_0x34b6[979]] * 2, true);
        this[_0x34b6[811]][_0x34b6[981]]();
        this[_0x34b6[811]][_0x34b6[982]]();
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
      local$$21951[_0x34b6[219]][_0x34b6[836]] = function(local$$22238, local$$22239, local$$22240, local$$22241, local$$22242, local$$22243) {
        this[_0x34b6[838]](local$$22238, local$$22239, local$$22240, local$$22241);
        this[_0x34b6[811]][_0x34b6[983]] = local$$22243.toString();
        this[_0x34b6[811]][_0x34b6[984]]();
      };
      /**
       * @param {?} local$$22278
       * @param {?} local$$22279
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[956]] = function(local$$22278, local$$22279) {
        this[_0x34b6[985]](local$$22278);
        this[_0x34b6[975]](local$$22279)[_0x34b6[982]]();
      };
      /**
       * @param {?} local$$22305
       * @return {?}
       */
      local$$21951[_0x34b6[219]][_0x34b6[986]] = function(local$$22305) {
        if (local$$22305[_0x34b6[588]] === null) {
          this[_0x34b6[971]][_0x34b6[542]](local$$22305[_0x34b6[554]], 0, 0);
          try {
            this[_0x34b6[971]][_0x34b6[401]](0, 0, 1, 1);
            /** @type {boolean} */
            local$$22305[_0x34b6[588]] = false;
          } catch (local$$22344) {
            this[_0x34b6[971]] = document[_0x34b6[424]](_0x34b6[516])[_0x34b6[403]](_0x34b6[402]);
            /** @type {boolean} */
            local$$22305[_0x34b6[588]] = true;
          }
        }
        return local$$22305[_0x34b6[588]];
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
      local$$21951[_0x34b6[219]][_0x34b6[542]] = function(local$$22389, local$$22390, local$$22391, local$$22392, local$$22393, local$$22394, local$$22395, local$$22396, local$$22397) {
        if (!this[_0x34b6[986]](local$$22389) || this[_0x34b6[494]][_0x34b6[497]]) {
          this[_0x34b6[811]][_0x34b6[542]](local$$22389[_0x34b6[554]], local$$22390, local$$22391, local$$22392, local$$22393, local$$22394, local$$22395, local$$22396, local$$22397);
        }
      };
      /**
       * @param {?} local$$22434
       * @param {?} local$$22435
       * @param {?} local$$22436
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[671]] = function(local$$22434, local$$22435, local$$22436) {
        this[_0x34b6[811]][_0x34b6[815]]();
        local$$22434[_0x34b6[618]](local$$22067)[_0x34b6[608]](function(local$$22453) {
          this[_0x34b6[985]](local$$22453)[_0x34b6[671]]();
        }, this);
        local$$22435[_0x34b6[238]](local$$22436);
        this[_0x34b6[811]][_0x34b6[810]]();
      };
      /**
       * @param {?} local$$22491
       * @return {?}
       */
      local$$21951[_0x34b6[219]][_0x34b6[985]] = function(local$$22491) {
        this[_0x34b6[811]][_0x34b6[978]]();
        local$$22491[_0x34b6[608]](function(local$$22504, local$$22505) {
          if (local$$22504[0] === _0x34b6[776]) {
            this[_0x34b6[811]][_0x34b6[776]][_0x34b6[652]](this[_0x34b6[811]], local$$22504[_0x34b6[388]](1));
          } else {
            this[_0x34b6[811]][local$$22505 === 0 ? _0x34b6[987] : local$$22504[0] + _0x34b6[988]][_0x34b6[652]](this[_0x34b6[811]], local$$22504[_0x34b6[388]](1));
          }
        }, this);
        this[_0x34b6[811]][_0x34b6[981]]();
        return this[_0x34b6[811]];
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
      local$$21951[_0x34b6[219]][_0x34b6[834]] = function(local$$22587, local$$22588, local$$22589, local$$22590, local$$22591, local$$22592) {
        this[_0x34b6[975]](local$$22587)[_0x34b6[834]] = [local$$22588, local$$22589, local$$22590, local$$22591, local$$22592][_0x34b6[2]](_0x34b6[711])[_0x34b6[379]](_0x34b6[477])[0];
      };
      /**
       * @param {string} local$$22628
       * @param {?} local$$22629
       * @param {?} local$$22630
       * @param {?} local$$22631
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[864]] = function(local$$22628, local$$22629, local$$22630, local$$22631) {
        this[_0x34b6[990]](_0x34b6[993], local$$22628.toString())[_0x34b6[990]](_0x34b6[992], local$$22629)[_0x34b6[990]](_0x34b6[991], local$$22630)[_0x34b6[990]](_0x34b6[989], local$$22631);
      };
      /**
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[865]] = function() {
        this[_0x34b6[990]](_0x34b6[993], _0x34b6[994]);
      };
      /**
       * @param {?} local$$22690
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[814]] = function(local$$22690) {
        this[_0x34b6[811]][_0x34b6[995]] = local$$22690;
      };
      /**
       * @param {?} local$$22710
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[816]] = function(local$$22710) {
        this[_0x34b6[811]][_0x34b6[996]](local$$22710[_0x34b6[602]][0], local$$22710[_0x34b6[602]][1]);
        this[_0x34b6[811]][_0x34b6[727]][_0x34b6[652]](this[_0x34b6[811]], local$$22710[_0x34b6[740]]);
        this[_0x34b6[811]][_0x34b6[996]](-local$$22710[_0x34b6[602]][0], -local$$22710[_0x34b6[602]][1]);
      };
      /**
       * @param {?} local$$22777
       * @param {?} local$$22778
       * @return {?}
       */
      local$$21951[_0x34b6[219]][_0x34b6[990]] = function(local$$22777, local$$22778) {
        if (this[_0x34b6[973]][local$$22777] !== local$$22778) {
          this[_0x34b6[973]][local$$22777] = this[_0x34b6[811]][local$$22777] = local$$22778;
        }
        return this;
      };
      /**
       * @param {?} local$$22810
       * @param {?} local$$22811
       * @param {?} local$$22812
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[739]] = function(local$$22810, local$$22811, local$$22812) {
        this[_0x34b6[811]][_0x34b6[997]](local$$22810, local$$22811, local$$22812);
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
      local$$21951[_0x34b6[219]][_0x34b6[962]] = function(local$$22832, local$$22833, local$$22834, local$$22835, local$$22836, local$$22837, local$$22838, local$$22839, local$$22840) {
        /** @type {!Array} */
        var local$$22891 = [[_0x34b6[896], Math[_0x34b6[292]](local$$22836), Math[_0x34b6[292]](local$$22837)], [_0x34b6[896], Math[_0x34b6[292]](local$$22836 + local$$22838), Math[_0x34b6[292]](local$$22837)], [_0x34b6[896], Math[_0x34b6[292]](local$$22836 + local$$22838), Math[_0x34b6[292]](local$$22839 + local$$22837)], [_0x34b6[896], Math[_0x34b6[292]](local$$22836), Math[_0x34b6[292]](local$$22839 + local$$22837)]];
        this[_0x34b6[671]]([local$$22891], function() {
          this[_0x34b6[967]](local$$22832, local$$22833, local$$22834, local$$22835, local$$22840[3], local$$22840[0]);
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
      local$$21951[_0x34b6[219]][_0x34b6[967]] = function(local$$22921, local$$22922, local$$22923, local$$22924, local$$22925, local$$22926) {
        var local$$22940 = Math[_0x34b6[292]](local$$22924[_0x34b6[432]] + local$$22922[_0x34b6[432]] + local$$22925);
        var local$$22954 = Math[_0x34b6[292]](local$$22924[_0x34b6[434]] + local$$22922[_0x34b6[434]] + local$$22926);
        this[_0x34b6[975]](this[_0x34b6[811]][_0x34b6[1E3]](this[_0x34b6[998]](local$$22921, local$$22923), _0x34b6[999]));
        this[_0x34b6[811]][_0x34b6[996]](local$$22940, local$$22954);
        this[_0x34b6[811]][_0x34b6[982]]();
        this[_0x34b6[811]][_0x34b6[996]](-local$$22940, -local$$22954);
      };
      /**
       * @param {?} local$$23010
       * @param {?} local$$23011
       * @return {undefined}
       */
      local$$21951[_0x34b6[219]][_0x34b6[959]] = function(local$$23010, local$$23011) {
        if (local$$23010 instanceof local$$22085) {
          var local$$23065 = this[_0x34b6[811]][_0x34b6[1001]](local$$23011[_0x34b6[432]] + local$$23011[_0x34b6[208]] * local$$23010[_0x34b6[582]], local$$23011[_0x34b6[434]] + local$$23011[_0x34b6[209]] * local$$23010[_0x34b6[583]], local$$23011[_0x34b6[432]] + local$$23011[_0x34b6[208]] * local$$23010[_0x34b6[584]], local$$23011[_0x34b6[434]] + local$$23011[_0x34b6[209]] * local$$23010[_0x34b6[585]]);
          local$$23010[_0x34b6[581]][_0x34b6[608]](function(local$$23073) {
            local$$23065[_0x34b6[1002]](local$$23073[_0x34b6[661]], local$$23073[_0x34b6[245]].toString());
          });
          this[_0x34b6[750]](local$$23011[_0x34b6[432]], local$$23011[_0x34b6[434]], local$$23011[_0x34b6[208]], local$$23011[_0x34b6[209]], local$$23065);
        }
      };
      /**
       * @param {?} local$$23121
       * @param {?} local$$23122
       * @return {?}
       */
      local$$21951[_0x34b6[219]][_0x34b6[998]] = function(local$$23121, local$$23122) {
        var local$$23127 = local$$23121[_0x34b6[554]];
        if (local$$23127[_0x34b6[208]] === local$$23122[_0x34b6[208]] && local$$23127[_0x34b6[209]] === local$$23122[_0x34b6[209]]) {
          return local$$23127;
        }
        var local$$23148;
        var local$$23156 = document[_0x34b6[424]](_0x34b6[516]);
        local$$23156[_0x34b6[208]] = local$$23122[_0x34b6[208]];
        local$$23156[_0x34b6[209]] = local$$23122[_0x34b6[209]];
        local$$23148 = local$$23156[_0x34b6[403]](_0x34b6[402]);
        local$$23148[_0x34b6[542]](local$$23127, 0, 0, local$$23127[_0x34b6[208]], local$$23127[_0x34b6[209]], 0, 0, local$$23122[_0x34b6[208]], local$$23122[_0x34b6[209]]);
        return local$$23156;
      };
      /** @type {function(?, ?): undefined} */
      local$$21948[_0x34b6[369]] = local$$21951;
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
        local$$23231[_0x34b6[238]](this, local$$23228, local$$23229);
        this[_0x34b6[1003]] = local$$23226;
        /** @type {!Array} */
        this[_0x34b6[794]] = [];
        /** @type {!Array} */
        this[_0x34b6[684]] = [];
        /** @type {number} */
        this[_0x34b6[322]] = (this[_0x34b6[667]] ? this[_0x34b6[667]][_0x34b6[668]][_0x34b6[322]] : 1) * local$$23227;
      }
      var local$$23231 = local$$23221(_0x34b6[486]);
      local$$23225[_0x34b6[219]] = Object[_0x34b6[242]](local$$23231[_0x34b6[219]]);
      /**
       * @param {?} local$$23298
       * @return {?}
       */
      local$$23225[_0x34b6[219]][_0x34b6[793]] = function(local$$23298) {
        var local$$23311 = this[_0x34b6[667]] ? this[_0x34b6[667]][_0x34b6[668]] : null;
        return local$$23311 ? local$$23311[_0x34b6[1003]] ? local$$23311 : local$$23311[_0x34b6[793]](local$$23298) : local$$23298[_0x34b6[668]];
      };
      /** @type {function(?, ?, ?, ?): undefined} */
      local$$23222[_0x34b6[369]] = local$$23225;
    }, {
      "./nodecontainer" : 14
    }],
    22 : [function(local$$23341, local$$23342, local$$23343) {
      /**
       * @param {?} local$$23346
       * @return {undefined}
       */
      function local$$23345(local$$23346) {
        this[_0x34b6[801]] = this[_0x34b6[1004]](local$$23346);
        this[_0x34b6[628]] = this[_0x34b6[1005]]();
        this[_0x34b6[613]] = this[_0x34b6[1006]]();
      }
      /**
       * @param {?} local$$23383
       * @return {?}
       */
      local$$23345[_0x34b6[219]][_0x34b6[1004]] = function(local$$23383) {
        var local$$23385;
        var local$$23387;
        var local$$23389;
        var local$$23391;
        /** @type {boolean} */
        var local$$23394 = false;
        if (local$$23383[_0x34b6[806]]) {
          local$$23385 = local$$23383[_0x34b6[806]]();
          if (local$$23385[_0x34b6[809]]) {
            local$$23387 = local$$23383[_0x34b6[424]](_0x34b6[1007]);
            local$$23387[_0x34b6[428]][_0x34b6[209]] = _0x34b6[1008];
            local$$23387[_0x34b6[428]][_0x34b6[687]] = _0x34b6[1009];
            local$$23383[_0x34b6[440]][_0x34b6[412]](local$$23387);
            local$$23385[_0x34b6[1010]](local$$23387);
            local$$23389 = local$$23385[_0x34b6[809]]();
            local$$23391 = local$$23389[_0x34b6[209]];
            if (local$$23391 === 123) {
              /** @type {boolean} */
              local$$23394 = true;
            }
            local$$23383[_0x34b6[440]][_0x34b6[529]](local$$23387);
          }
        }
        return local$$23394;
      };
      /**
       * @return {?}
       */
      local$$23345[_0x34b6[219]][_0x34b6[1005]] = function() {
        return typeof(new Image)[_0x34b6[589]] !== _0x34b6[367];
      };
      /**
       * @return {?}
       */
      local$$23345[_0x34b6[219]][_0x34b6[1006]] = function() {
        /** @type {!Image} */
        var local$$23514 = new Image;
        var local$$23522 = document[_0x34b6[424]](_0x34b6[516]);
        var local$$23530 = local$$23522[_0x34b6[403]](_0x34b6[402]);
        local$$23514[_0x34b6[551]] = _0x34b6[1011];
        try {
          local$$23530[_0x34b6[542]](local$$23514, 0, 0);
          local$$23522[_0x34b6[1012]]();
        } catch (local$$23552) {
          return false;
        }
        return true;
      };
      /** @type {function(?): undefined} */
      local$$23342[_0x34b6[369]] = local$$23345;
    }, {}],
    23 : [function(local$$23576, local$$23577, local$$23578) {
      /**
       * @param {?} local$$23581
       * @return {undefined}
       */
      function local$$23580(local$$23581) {
        this[_0x34b6[551]] = local$$23581;
        /** @type {null} */
        this[_0x34b6[554]] = null;
        var local$$23594 = this;
        this[_0x34b6[553]] = this[_0x34b6[1017]]()[_0x34b6[507]](function() {
          return local$$23594[_0x34b6[632]](local$$23581) ? Promise[_0x34b6[526]](local$$23594[_0x34b6[1016]](local$$23581)) : local$$23619(local$$23581);
        })[_0x34b6[507]](function(local$$23629) {
          return new Promise(function(local$$23631) {
            window[_0x34b6[518]][_0x34b6[613]][_0x34b6[1015]][_0x34b6[1014]](local$$23629, local$$23594[_0x34b6[1013]][_0x34b6[238]](local$$23594, local$$23631));
          });
        });
      }
      var local$$23619 = local$$23576(_0x34b6[922]);
      var local$$23675 = local$$23576(_0x34b6[487])[_0x34b6[923]];
      /**
       * @return {?}
       */
      local$$23580[_0x34b6[219]][_0x34b6[1017]] = function() {
        return !window[_0x34b6[518]][_0x34b6[613]] || !window[_0x34b6[518]][_0x34b6[613]][_0x34b6[1015]] ? Promise[_0x34b6[504]](new Error(_0x34b6[1018])) : Promise[_0x34b6[526]]();
      };
      /**
       * @param {?} local$$23725
       * @return {?}
       */
      local$$23580[_0x34b6[219]][_0x34b6[1016]] = function(local$$23725) {
        return /^data:image\/svg\+xml;base64,/[_0x34b6[386]](local$$23725) ? this[_0x34b6[923]](this[_0x34b6[1019]](local$$23725)) : this[_0x34b6[1019]](local$$23725);
      };
      /**
       * @param {?} local$$23757
       * @return {?}
       */
      local$$23580[_0x34b6[219]][_0x34b6[1019]] = function(local$$23757) {
        return local$$23757[_0x34b6[626]](/^data:image\/svg\+xml(;base64)?,/, _0x34b6[381]);
      };
      /**
       * @param {?} local$$23778
       * @return {?}
       */
      local$$23580[_0x34b6[219]][_0x34b6[632]] = function(local$$23778) {
        return /^data:image\/svg\+xml/i[_0x34b6[386]](local$$23778);
      };
      /**
       * @param {?} local$$23797
       * @return {?}
       */
      local$$23580[_0x34b6[219]][_0x34b6[1013]] = function(local$$23797) {
        var local$$23799 = this;
        return function(local$$23801, local$$23802) {
          var local$$23818 = new window[_0x34b6[518]][_0x34b6[613]][_0x34b6[1015]].StaticCanvas(_0x34b6[1020]);
          local$$23799[_0x34b6[554]] = local$$23818[_0x34b6[1021]];
          local$$23818[_0x34b6[1026]](local$$23802[_0x34b6[208]])[_0x34b6[1025]](local$$23802[_0x34b6[209]])[_0x34b6[274]](window[_0x34b6[518]][_0x34b6[613]][_0x34b6[1015]][_0x34b6[1024]][_0x34b6[1023]](local$$23801, local$$23802))[_0x34b6[1022]]();
          local$$23797(local$$23818[_0x34b6[1021]]);
        };
      };
      /**
       * @param {?} local$$23885
       * @return {?}
       */
      local$$23580[_0x34b6[219]][_0x34b6[923]] = function(local$$23885) {
        return typeof window[_0x34b6[1027]] === _0x34b6[391] ? window[_0x34b6[1027]](local$$23885) : local$$23675(local$$23885);
      };
      /** @type {function(?): undefined} */
      local$$23577[_0x34b6[369]] = local$$23580;
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
        this[_0x34b6[551]] = local$$23922;
        /** @type {null} */
        this[_0x34b6[554]] = null;
        var local$$23936 = this;
        this[_0x34b6[553]] = local$$23923 ? new Promise(function(local$$23941, local$$23942) {
          /** @type {!Image} */
          local$$23936[_0x34b6[554]] = new Image;
          local$$23936[_0x34b6[554]][_0x34b6[443]] = local$$23941;
          local$$23936[_0x34b6[554]][_0x34b6[556]] = local$$23942;
          local$$23936[_0x34b6[554]][_0x34b6[551]] = _0x34b6[1028] + (new XMLSerializer)[_0x34b6[1029]](local$$23922);
          if (local$$23936[_0x34b6[554]][_0x34b6[557]] === true) {
            local$$23941(local$$23936[_0x34b6[554]]);
          }
        }) : this[_0x34b6[1017]]()[_0x34b6[507]](function() {
          return new Promise(function(local$$24009) {
            window[_0x34b6[518]][_0x34b6[613]][_0x34b6[1015]][_0x34b6[1030]](local$$23922, local$$23936[_0x34b6[1013]][_0x34b6[238]](local$$23936, local$$24009));
          });
        });
      }
      var local$$24047 = local$$23917(_0x34b6[595]);
      local$$23921[_0x34b6[219]] = Object[_0x34b6[242]](local$$24047[_0x34b6[219]]);
      /** @type {function(?, !Object): undefined} */
      local$$23918[_0x34b6[369]] = local$$23921;
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
        local$$24080[_0x34b6[238]](this, local$$24077, local$$24078);
      }
      /**
       * @param {?} local$$24089
       * @param {?} local$$24090
       * @param {?} local$$24091
       * @return {?}
       */
      function local$$24088(local$$24089, local$$24090, local$$24091) {
        if (local$$24089[_0x34b6[223]] > 0) {
          return local$$24090 + local$$24091[_0x34b6[701]]();
        }
      }
      var local$$24080 = local$$24072(_0x34b6[486]);
      local$$24076[_0x34b6[219]] = Object[_0x34b6[242]](local$$24080[_0x34b6[219]]);
      /**
       * @return {undefined}
       */
      local$$24076[_0x34b6[219]][_0x34b6[854]] = function() {
        this[_0x34b6[609]][_0x34b6[575]] = this[_0x34b6[727]](this[_0x34b6[667]][_0x34b6[688]](_0x34b6[1031]));
      };
      /**
       * @param {?} local$$24162
       * @return {?}
       */
      local$$24076[_0x34b6[219]][_0x34b6[727]] = function(local$$24162) {
        var local$$24170 = this[_0x34b6[609]][_0x34b6[575]];
        switch(local$$24162) {
          case _0x34b6[1032]:
            return local$$24170[_0x34b6[387]]();
          case _0x34b6[1033]:
            return local$$24170[_0x34b6[626]](/(^|\s|:|-|\(|\))([a-z])/g, local$$24088);
          case _0x34b6[1034]:
            return local$$24170[_0x34b6[701]]();
          default:
            return local$$24170;
        }
      };
      /** @type {function(?, ?): undefined} */
      local$$24073[_0x34b6[369]] = local$$24076;
    }, {
      "./nodecontainer" : 14
    }],
    26 : [function(local$$24220, local$$24221, local$$24222) {
      /**
       * @return {?}
       */
      local$$24222[_0x34b6[550]] = function local$$24227() {
        return _0x34b6[1035];
      };
      /**
       * @param {?} local$$24239
       * @param {?} local$$24240
       * @return {?}
       */
      local$$24222[_0x34b6[599]] = function(local$$24239, local$$24240) {
        return function() {
          return local$$24239[_0x34b6[652]](local$$24240, arguments);
        };
      };
      /**
       * @param {!Object} local$$24258
       * @return {?}
       */
      local$$24222[_0x34b6[923]] = function(local$$24258) {
        var local$$24262 = _0x34b6[1036];
        var local$$24267 = local$$24258[_0x34b6[223]];
        var local$$24269;
        var local$$24271;
        var local$$24273;
        var local$$24275;
        var local$$24277;
        var local$$24279;
        var local$$24281;
        var local$$24283;
        var local$$24287 = _0x34b6[381];
        /** @type {number} */
        local$$24269 = 0;
        for (; local$$24269 < local$$24267; local$$24269 = local$$24269 + 4) {
          local$$24271 = local$$24262[_0x34b6[742]](local$$24258[local$$24269]);
          local$$24273 = local$$24262[_0x34b6[742]](local$$24258[local$$24269 + 1]);
          local$$24275 = local$$24262[_0x34b6[742]](local$$24258[local$$24269 + 2]);
          local$$24277 = local$$24262[_0x34b6[742]](local$$24258[local$$24269 + 3]);
          /** @type {number} */
          local$$24279 = local$$24271 << 2 | local$$24273 >> 4;
          /** @type {number} */
          local$$24281 = (local$$24273 & 15) << 4 | local$$24275 >> 2;
          /** @type {number} */
          local$$24283 = (local$$24275 & 3) << 6 | local$$24277;
          if (local$$24275 === 64) {
            local$$24287 = local$$24287 + String[_0x34b6[377]](local$$24279);
          } else {
            if (local$$24277 === 64 || local$$24277 === -1) {
              local$$24287 = local$$24287 + String[_0x34b6[377]](local$$24279, local$$24281);
            } else {
              local$$24287 = local$$24287 + String[_0x34b6[377]](local$$24279, local$$24281, local$$24283);
            }
          }
        }
        return local$$24287;
      };
      /**
       * @param {?} local$$24399
       * @return {?}
       */
      local$$24222[_0x34b6[491]] = function(local$$24399) {
        if (local$$24399[_0x34b6[809]]) {
          var local$$24408 = local$$24399[_0x34b6[809]]();
          var local$$24422 = local$$24399[_0x34b6[544]] == null ? local$$24408[_0x34b6[208]] : local$$24399[_0x34b6[544]];
          return {
            top : local$$24408[_0x34b6[434]],
            bottom : local$$24408[_0x34b6[656]] || local$$24408[_0x34b6[434]] + local$$24408[_0x34b6[209]],
            right : local$$24408[_0x34b6[432]] + local$$24422,
            left : local$$24408[_0x34b6[432]],
            width : local$$24422,
            height : local$$24399[_0x34b6[547]] == null ? local$$24408[_0x34b6[209]] : local$$24399[_0x34b6[547]]
          };
        }
        return {};
      };
      /**
       * @param {?} local$$24471
       * @return {?}
       */
      local$$24222[_0x34b6[666]] = function(local$$24471) {
        var local$$24487 = local$$24471[_0x34b6[1037]] ? local$$24222[_0x34b6[666]](local$$24471[_0x34b6[1037]]) : {
          top : 0,
          left : 0
        };
        return {
          top : local$$24471[_0x34b6[568]] + local$$24487[_0x34b6[434]],
          bottom : local$$24471[_0x34b6[568]] + local$$24471[_0x34b6[547]] + local$$24487[_0x34b6[434]],
          right : local$$24471[_0x34b6[1038]] + local$$24487[_0x34b6[432]] + local$$24471[_0x34b6[544]],
          left : local$$24471[_0x34b6[1038]] + local$$24487[_0x34b6[432]],
          width : local$$24471[_0x34b6[544]],
          height : local$$24471[_0x34b6[547]]
        };
      };
      /**
       * @param {?} local$$24540
       * @return {?}
       */
      local$$24222[_0x34b6[665]] = function(local$$24540) {
        var local$$24544 = _0x34b6[1039];
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
            if (local$$24548[_0x34b6[702]](0, 1) === _0x34b6[1040]) {
              local$$24548 = local$$24548[_0x34b6[702]](1, local$$24548[_0x34b6[223]] - 2);
            }
            if (local$$24548) {
              local$$24567[_0x34b6[220]](local$$24548);
            }
            if (local$$24546[_0x34b6[702]](0, 1) === _0x34b6[372] && (local$$24552 = local$$24546[_0x34b6[742]](_0x34b6[372], 1) + 1) > 0) {
              local$$24550 = local$$24546[_0x34b6[702]](0, local$$24552);
              local$$24546 = local$$24546[_0x34b6[702]](local$$24552);
            }
            local$$24557[_0x34b6[220]]({
              prefix : local$$24550,
              method : local$$24546[_0x34b6[387]](),
              value : local$$24554,
              args : local$$24567,
              image : null
            });
          }
          /** @type {!Array} */
          local$$24567 = [];
          local$$24546 = local$$24550 = local$$24548 = local$$24554 = _0x34b6[381];
        };
        /** @type {!Array} */
        local$$24567 = [];
        local$$24546 = local$$24550 = local$$24548 = local$$24554 = _0x34b6[381];
        local$$24540[_0x34b6[379]](_0x34b6[381])[_0x34b6[608]](function(local$$24688) {
          if (local$$24560 === 0 && local$$24544[_0x34b6[742]](local$$24688) > -1) {
            return;
          }
          switch(local$$24688) {
            case _0x34b6[1040]:
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
            case _0x34b6[1041]:
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
            case _0x34b6[478]:
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
            case _0x34b6[477]:
              if (local$$24565) {
                break;
              } else {
                if (local$$24560 === 0) {
                  local$$24667();
                  return;
                } else {
                  if (local$$24560 === 1) {
                    if (local$$24563 === 0 && !local$$24546[_0x34b6[473]](/^url$/i)) {
                      local$$24567[_0x34b6[220]](local$$24548);
                      local$$24548 = _0x34b6[381];
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
        local$$24861[_0x34b6[652]](this, arguments);
        this[_0x34b6[445]] = local$$24859[_0x34b6[622]][0] === _0x34b6[1042] ? local$$24861[_0x34b6[586]][_0x34b6[653]] : local$$24861[_0x34b6[586]][_0x34b6[1043]];
      }
      var local$$24861 = local$$24854(_0x34b6[650]);
      local$$24858[_0x34b6[219]] = Object[_0x34b6[242]](local$$24861[_0x34b6[219]]);
      /** @type {function(?): undefined} */
      local$$24855[_0x34b6[369]] = local$$24858;
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
          local$$24933[_0x34b6[452]](_0x34b6[1044], local$$24927);
          /**
           * @return {undefined}
           */
          local$$24933[_0x34b6[443]] = function() {
            if (local$$24933[_0x34b6[1045]] === 200) {
              local$$24929(local$$24933[_0x34b6[1046]]);
            } else {
              local$$24930(new Error(local$$24933[_0x34b6[1047]]));
            }
          };
          /**
           * @return {undefined}
           */
          local$$24933[_0x34b6[556]] = function() {
            local$$24930(new Error(_0x34b6[1048]));
          };
          local$$24933[_0x34b6[1049]]();
        });
      }
      /** @type {function(?): ?} */
      local$$24923[_0x34b6[369]] = local$$24926;
    }, {}]
  }, {}, [4])(4);
});
/**
 * @param {string} local$$25021
 * @return {undefined}
 */
THREE[_0x34b6[1054]] = function(local$$25021) {
  this[_0x34b6[1055]] = local$$25021 !== undefined ? local$$25021 : THREE[_0x34b6[1056]];
};
Object[_0x34b6[233]](THREE[_0x34b6[1054]][_0x34b6[219]], THREE[_0x34b6[1057]][_0x34b6[219]], {
  load : function(local$$25052, local$$25053, local$$25054, local$$25055) {
    var local$$25057 = this;
    var local$$25065 = new THREE.XHRLoader(this[_0x34b6[1055]]);
    local$$25065[_0x34b6[1059]](this[_0x34b6[1058]]);
    local$$25065[_0x34b6[1060]](local$$25052, function(local$$25078) {
      local$$25053(local$$25057[_0x34b6[768]](local$$25078));
    }, local$$25054, local$$25055);
  },
  setPath : function(local$$25092) {
    this[_0x34b6[1058]] = local$$25092;
  },
  setTexturePath : function(local$$25101) {
    this[_0x34b6[1061]] = local$$25101;
  },
  setBaseUrl : function(local$$25110) {
    console[_0x34b6[1063]](_0x34b6[1062]);
    this[_0x34b6[1064]](local$$25110);
  },
  setCrossOrigin : function(local$$25127) {
    this[_0x34b6[589]] = local$$25127;
  },
  setMaterialOptions : function(local$$25136) {
    this[_0x34b6[1065]] = local$$25136;
  },
  parse : function(local$$25146) {
    var local$$25154 = local$$25146[_0x34b6[379]](_0x34b6[1]);
    var local$$25157 = {};
    /** @type {!RegExp} */
    var local$$25160 = /\s+/;
    var local$$25163 = {};
    /** @type {number} */
    var local$$25166 = 0;
    for (; local$$25166 < local$$25154[_0x34b6[223]]; local$$25166++) {
      var local$$25175 = local$$25154[local$$25166];
      local$$25175 = local$$25175[_0x34b6[712]]();
      if (local$$25175[_0x34b6[223]] === 0 || local$$25175[_0x34b6[1066]](0) === _0x34b6[1067]) {
        continue;
      }
      var local$$25207 = local$$25175[_0x34b6[742]](_0x34b6[711]);
      var local$$25217 = local$$25207 >= 0 ? local$$25175[_0x34b6[474]](0, local$$25207) : local$$25175;
      local$$25217 = local$$25217[_0x34b6[387]]();
      var local$$25236 = local$$25207 >= 0 ? local$$25175[_0x34b6[474]](local$$25207 + 1) : _0x34b6[381];
      local$$25236 = local$$25236[_0x34b6[712]]();
      if (local$$25217 === _0x34b6[1068]) {
        local$$25157 = {
          name : local$$25236
        };
        local$$25163[local$$25236] = local$$25157;
      } else {
        if (local$$25157) {
          if (local$$25217 === _0x34b6[1069] || local$$25217 === _0x34b6[1070] || local$$25217 === _0x34b6[1071]) {
            var local$$25270 = local$$25236[_0x34b6[379]](local$$25160, 3);
            /** @type {!Array} */
            local$$25157[local$$25217] = [parseFloat(local$$25270[0]), parseFloat(local$$25270[1]), parseFloat(local$$25270[2])];
          } else {
            local$$25157[local$$25217] = local$$25236;
          }
        }
      }
    }
    var local$$25313 = new THREE[_0x34b6[1054]].MaterialCreator(this[_0x34b6[1061]] || this[_0x34b6[1058]], this[_0x34b6[1065]]);
    local$$25313[_0x34b6[1072]](this[_0x34b6[589]]);
    local$$25313[_0x34b6[1073]](this[_0x34b6[1055]]);
    local$$25313[_0x34b6[1074]](local$$25163);
    return local$$25313;
  }
});
/**
 * @param {?} local$$25348
 * @param {?} local$$25349
 * @return {undefined}
 */
THREE[_0x34b6[1054]][_0x34b6[1075]] = function(local$$25348, local$$25349) {
  this[_0x34b6[1076]] = local$$25348 || _0x34b6[381];
  this[_0x34b6[494]] = local$$25349;
  this[_0x34b6[1077]] = {};
  this[_0x34b6[1078]] = {};
  /** @type {!Array} */
  this[_0x34b6[1079]] = [];
  this[_0x34b6[1080]] = {};
  this[_0x34b6[294]] = this[_0x34b6[494]] && this[_0x34b6[494]][_0x34b6[294]] ? this[_0x34b6[494]][_0x34b6[294]] : THREE[_0x34b6[1081]];
  this[_0x34b6[1082]] = this[_0x34b6[494]] && this[_0x34b6[494]][_0x34b6[1082]] ? this[_0x34b6[494]][_0x34b6[1082]] : THREE[_0x34b6[1083]];
};
THREE[_0x34b6[1054]][_0x34b6[1075]][_0x34b6[219]] = {
  constructor : THREE[_0x34b6[1054]][_0x34b6[1075]],
  setCrossOrigin : function(local$$25457) {
    this[_0x34b6[589]] = local$$25457;
  },
  setManager : function(local$$25466) {
    this[_0x34b6[1055]] = local$$25466;
  },
  setMaterials : function(local$$25475) {
    this[_0x34b6[1077]] = this[_0x34b6[1084]](local$$25475);
    this[_0x34b6[1078]] = {};
    /** @type {!Array} */
    this[_0x34b6[1079]] = [];
    this[_0x34b6[1080]] = {};
  },
  convert : function(local$$25506) {
    if (!this[_0x34b6[494]]) {
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
        var local$$25542 = local$$25530[_0x34b6[387]]();
        switch(local$$25542) {
          case _0x34b6[1070]:
          case _0x34b6[1069]:
          case _0x34b6[1071]:
            if (this[_0x34b6[494]] && this[_0x34b6[494]][_0x34b6[1085]]) {
              /** @type {!Array} */
              local$$25536 = [local$$25536[0] / 255, local$$25536[1] / 255, local$$25536[2] / 255];
            }
            if (this[_0x34b6[494]] && this[_0x34b6[494]][_0x34b6[1086]]) {
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
    for (local$$25638 in this[_0x34b6[1077]]) {
      this[_0x34b6[242]](local$$25638);
    }
  },
  getIndex : function(local$$25652) {
    return this[_0x34b6[1080]][local$$25652];
  },
  getAsArray : function() {
    /** @type {number} */
    var local$$25664 = 0;
    var local$$25666;
    for (local$$25666 in this[_0x34b6[1077]]) {
      this[_0x34b6[1079]][local$$25664] = this[_0x34b6[242]](local$$25666);
      /** @type {number} */
      this[_0x34b6[1080]][local$$25666] = local$$25664;
      local$$25664++;
    }
    return this[_0x34b6[1079]];
  },
  create : function(local$$25699) {
    if (this[_0x34b6[1078]][local$$25699] === undefined) {
      this[_0x34b6[1087]](local$$25699);
    }
    return this[_0x34b6[1078]][local$$25699];
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
      var local$$25738 = local$$25733[_0x34b6[1088]](local$$25725, local$$25727);
      var local$$25752 = local$$25733[_0x34b6[1089]](local$$25743(local$$25733[_0x34b6[1076]], local$$25738[_0x34b6[610]]));
      local$$25752[_0x34b6[999]][_0x34b6[338]](local$$25738[_0x34b6[1090]]);
      local$$25752[_0x34b6[1091]][_0x34b6[338]](local$$25738[_0x34b6[1091]]);
      local$$25752[_0x34b6[1092]] = local$$25733[_0x34b6[1082]];
      local$$25752[_0x34b6[1093]] = local$$25733[_0x34b6[1082]];
      local$$25727[local$$25724] = local$$25752;
    }
    var local$$25733 = this;
    var local$$25802 = this[_0x34b6[1077]][local$$25721];
    var local$$25727 = {
      name : local$$25721,
      side : this[_0x34b6[294]]
    };
    /**
     * @param {(Object|number)} local$$25809
     * @param {!Object} local$$25810
     * @return {?}
     */
    var local$$25743 = function(local$$25809, local$$25810) {
      if (typeof local$$25810 !== _0x34b6[501] || local$$25810 === _0x34b6[381]) {
        return _0x34b6[381];
      }
      if (/^https?:\/\//i[_0x34b6[386]](local$$25810)) {
        return local$$25810;
      }
      return local$$25809 + local$$25810;
    };
    var local$$25841;
    for (local$$25841 in local$$25802) {
      var local$$25844 = local$$25802[local$$25841];
      if (local$$25844 === _0x34b6[381]) {
        continue;
      }
      switch(local$$25841[_0x34b6[387]]()) {
        case _0x34b6[1070]:
          local$$25727[_0x34b6[245]] = (new THREE.Color)[_0x34b6[462]](local$$25844);
          break;
        case _0x34b6[1071]:
          local$$25727[_0x34b6[1094]] = (new THREE.Color)[_0x34b6[462]](local$$25844);
          break;
        case _0x34b6[1095]:
          local$$25723(_0x34b6[645], local$$25844);
          break;
        case _0x34b6[1097]:
          local$$25723(_0x34b6[1096], local$$25844);
          break;
        case _0x34b6[1098]:
        case _0x34b6[1100]:
          local$$25723(_0x34b6[1099], local$$25844);
          break;
        case _0x34b6[1102]:
          /** @type {number} */
          local$$25727[_0x34b6[1101]] = parseFloat(local$$25844);
          break;
        case _0x34b6[1103]:
          if (local$$25844 < 1) {
            local$$25727[_0x34b6[322]] = local$$25844;
            /** @type {boolean} */
            local$$25727[_0x34b6[480]] = true;
          }
          break;
        case _0x34b6[1104]:
          if (local$$25844 > 0) {
            /** @type {number} */
            local$$25727[_0x34b6[322]] = 1 - local$$25844;
            /** @type {boolean} */
            local$$25727[_0x34b6[480]] = true;
          }
          break;
        default:
          break;
      }
    }
    this[_0x34b6[1078]][local$$25721] = new THREE.MeshPhongMaterial(local$$25727);
    return this[_0x34b6[1078]][local$$25721];
  },
  getTextureParams : function(local$$25999, local$$26000) {
    var local$$26011 = {
      scale : new THREE.Vector2(1, 1),
      offset : new THREE.Vector2(0, 0)
    };
    var local$$26018 = local$$25999[_0x34b6[379]](/\s+/);
    var local$$26020;
    local$$26020 = local$$26018[_0x34b6[742]](_0x34b6[1105]);
    if (local$$26020 >= 0) {
      /** @type {number} */
      local$$26000[_0x34b6[1106]] = parseFloat(local$$26018[local$$26020 + 1]);
      local$$26018[_0x34b6[222]](local$$26020, 2);
    }
    local$$26020 = local$$26018[_0x34b6[742]](_0x34b6[1107]);
    if (local$$26020 >= 0) {
      local$$26011[_0x34b6[1090]][_0x34b6[334]](parseFloat(local$$26018[local$$26020 + 1]), parseFloat(local$$26018[local$$26020 + 2]));
      local$$26018[_0x34b6[222]](local$$26020, 4);
    }
    local$$26020 = local$$26018[_0x34b6[742]](_0x34b6[1108]);
    if (local$$26020 >= 0) {
      local$$26011[_0x34b6[1091]][_0x34b6[334]](parseFloat(local$$26018[local$$26020 + 1]), parseFloat(local$$26018[local$$26020 + 2]));
      local$$26018[_0x34b6[222]](local$$26020, 4);
    }
    local$$26011[_0x34b6[610]] = local$$26018[_0x34b6[2]](_0x34b6[711])[_0x34b6[712]]();
    return local$$26011;
  },
  loadTexture : function(local$$26138, local$$26139, local$$26140, local$$26141, local$$26142) {
    var local$$26144;
    var local$$26156 = THREE[_0x34b6[1110]][_0x34b6[1109]][_0x34b6[640]](local$$26138);
    var local$$26169 = this[_0x34b6[1055]] !== undefined ? this[_0x34b6[1055]] : THREE[_0x34b6[1056]];
    if (local$$26156 === null) {
      local$$26156 = new THREE.TextureLoader(local$$26169);
    }
    if (local$$26156[_0x34b6[1072]]) {
      local$$26156[_0x34b6[1072]](this[_0x34b6[589]]);
    }
    local$$26144 = local$$26156[_0x34b6[1060]](local$$26138, local$$26140, local$$26141, local$$26142);
    if (local$$26139 !== undefined) {
      /** @type {string} */
      local$$26144[_0x34b6[1111]] = local$$26139;
    }
    return local$$26144;
  }
};
/**
 * @param {string} local$$26221
 * @return {undefined}
 */
THREE[_0x34b6[1112]] = function(local$$26221) {
  this[_0x34b6[1055]] = local$$26221 !== undefined ? local$$26221 : THREE[_0x34b6[1056]];
  /** @type {null} */
  this[_0x34b6[1078]] = null;
  this[_0x34b6[1113]] = {
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
THREE[_0x34b6[1112]][_0x34b6[219]] = {
  constructor : THREE[_0x34b6[1112]],
  load : function(local$$26280, local$$26281, local$$26282, local$$26283) {
    var local$$26285 = this;
    var local$$26292 = new THREE.XHRLoader(local$$26285[_0x34b6[1055]]);
    local$$26292[_0x34b6[1059]](this[_0x34b6[1058]]);
    local$$26292[_0x34b6[1060]](local$$26280, function(local$$26305) {
      local$$26281(local$$26285[_0x34b6[768]](local$$26305));
    }, local$$26282, local$$26283);
  },
  setPath : function(local$$26319) {
    this[_0x34b6[1058]] = local$$26319;
  },
  setMaterials : function(local$$26328) {
    this[_0x34b6[1078]] = local$$26328;
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
        if (this[_0x34b6[368]] && this[_0x34b6[368]][_0x34b6[1114]] === false) {
          /** @type {string} */
          this[_0x34b6[368]][_0x34b6[1115]] = local$$26344;
          /** @type {boolean} */
          this[_0x34b6[368]][_0x34b6[1114]] = local$$26345 !== false;
          return;
        }
        var local$$26403 = this[_0x34b6[368]] && typeof this[_0x34b6[368]][_0x34b6[1116]] === _0x34b6[391] ? this[_0x34b6[368]][_0x34b6[1116]]() : undefined;
        if (this[_0x34b6[368]] && typeof this[_0x34b6[368]][_0x34b6[1117]] === _0x34b6[391]) {
          this[_0x34b6[368]]._finalize(true);
        }
        this[_0x34b6[368]] = {
          name : local$$26344 || _0x34b6[381],
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
            if (local$$26449 && (local$$26449[_0x34b6[1118]] || local$$26449[_0x34b6[1119]] <= 0)) {
              this[_0x34b6[1078]][_0x34b6[222]](local$$26449[_0x34b6[1120]], 1);
            }
            var local$$26563 = {
              index : this[_0x34b6[1078]][_0x34b6[223]],
              name : local$$26443 || _0x34b6[381],
              mtllib : Array[_0x34b6[471]](local$$26444) && local$$26444[_0x34b6[223]] > 0 ? local$$26444[local$$26444[_0x34b6[223]] - 1] : _0x34b6[381],
              smooth : local$$26449 !== undefined ? local$$26449[_0x34b6[1121]] : this[_0x34b6[1121]],
              groupStart : local$$26449 !== undefined ? local$$26449[_0x34b6[1122]] : 0,
              groupEnd : -1,
              groupCount : -1,
              inherited : false,
              clone : function(local$$26521) {
                var local$$26545 = {
                  index : typeof local$$26521 === _0x34b6[1123] ? local$$26521 : this[_0x34b6[1120]],
                  name : this[_0x34b6[1115]],
                  mtllib : this[_0x34b6[1124]],
                  smooth : this[_0x34b6[1121]],
                  groupStart : 0,
                  groupEnd : -1,
                  groupCount : -1,
                  inherited : false
                };
                local$$26545[_0x34b6[212]] = this[_0x34b6[212]][_0x34b6[599]](local$$26545);
                return local$$26545;
              }
            };
            this[_0x34b6[1078]][_0x34b6[220]](local$$26563);
            return local$$26563;
          },
          currentMaterial : function() {
            if (this[_0x34b6[1078]][_0x34b6[223]] > 0) {
              return this[_0x34b6[1078]][this[_0x34b6[1078]][_0x34b6[223]] - 1];
            }
            return undefined;
          },
          _finalize : function(local$$26604) {
            var local$$26610 = this[_0x34b6[1116]]();
            if (local$$26610 && local$$26610[_0x34b6[1122]] === -1) {
              /** @type {number} */
              local$$26610[_0x34b6[1122]] = this[_0x34b6[1126]][_0x34b6[1125]][_0x34b6[223]] / 3;
              /** @type {number} */
              local$$26610[_0x34b6[1119]] = local$$26610[_0x34b6[1122]] - local$$26610[_0x34b6[1127]];
              /** @type {boolean} */
              local$$26610[_0x34b6[1118]] = false;
            }
            if (local$$26604 && this[_0x34b6[1078]][_0x34b6[223]] > 1) {
              /** @type {number} */
              var local$$26672 = this[_0x34b6[1078]][_0x34b6[223]] - 1;
              for (; local$$26672 >= 0; local$$26672--) {
                if (this[_0x34b6[1078]][local$$26672][_0x34b6[1119]] <= 0) {
                  this[_0x34b6[1078]][_0x34b6[222]](local$$26672, 1);
                }
              }
            }
            if (local$$26604 && this[_0x34b6[1078]][_0x34b6[223]] === 0) {
              this[_0x34b6[1078]][_0x34b6[220]]({
                name : _0x34b6[381],
                smooth : this[_0x34b6[1121]]
              });
            }
            return local$$26610;
          }
        };
        if (local$$26403 && local$$26403[_0x34b6[1115]] && typeof local$$26403[_0x34b6[212]] === _0x34b6[391]) {
          var local$$26752 = local$$26403[_0x34b6[212]](0);
          /** @type {boolean} */
          local$$26752[_0x34b6[1118]] = true;
          this[_0x34b6[368]][_0x34b6[1078]][_0x34b6[220]](local$$26752);
        }
        this[_0x34b6[1128]][_0x34b6[220]](this[_0x34b6[368]]);
      },
      finalize : function() {
        if (this[_0x34b6[368]] && typeof this[_0x34b6[368]][_0x34b6[1117]] === _0x34b6[391]) {
          this[_0x34b6[368]]._finalize(true);
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
        var local$$26880 = this[_0x34b6[1125]];
        var local$$26891 = this[_0x34b6[368]][_0x34b6[1126]][_0x34b6[1125]];
        local$$26891[_0x34b6[220]](local$$26880[local$$26873 + 0]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26873 + 1]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26873 + 2]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26874 + 0]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26874 + 1]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26874 + 2]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26875 + 0]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26875 + 1]);
        local$$26891[_0x34b6[220]](local$$26880[local$$26875 + 2]);
      },
      addVertexLine : function(local$$26967) {
        var local$$26972 = this[_0x34b6[1125]];
        var local$$26983 = this[_0x34b6[368]][_0x34b6[1126]][_0x34b6[1125]];
        local$$26983[_0x34b6[220]](local$$26972[local$$26967 + 0]);
        local$$26983[_0x34b6[220]](local$$26972[local$$26967 + 1]);
        local$$26983[_0x34b6[220]](local$$26972[local$$26967 + 2]);
      },
      addNormal : function(local$$27011, local$$27012, local$$27013) {
        var local$$27018 = this[_0x34b6[1129]];
        var local$$27029 = this[_0x34b6[368]][_0x34b6[1126]][_0x34b6[1129]];
        local$$27029[_0x34b6[220]](local$$27018[local$$27011 + 0]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27011 + 1]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27011 + 2]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27012 + 0]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27012 + 1]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27012 + 2]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27013 + 0]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27013 + 1]);
        local$$27029[_0x34b6[220]](local$$27018[local$$27013 + 2]);
      },
      addUV : function(local$$27105, local$$27106, local$$27107) {
        var local$$27112 = this[_0x34b6[1130]];
        var local$$27123 = this[_0x34b6[368]][_0x34b6[1126]][_0x34b6[1130]];
        local$$27123[_0x34b6[220]](local$$27112[local$$27105 + 0]);
        local$$27123[_0x34b6[220]](local$$27112[local$$27105 + 1]);
        local$$27123[_0x34b6[220]](local$$27112[local$$27106 + 0]);
        local$$27123[_0x34b6[220]](local$$27112[local$$27106 + 1]);
        local$$27123[_0x34b6[220]](local$$27112[local$$27107 + 0]);
        local$$27123[_0x34b6[220]](local$$27112[local$$27107 + 1]);
      },
      addUVLine : function(local$$27175) {
        var local$$27180 = this[_0x34b6[1130]];
        var local$$27191 = this[_0x34b6[368]][_0x34b6[1126]][_0x34b6[1130]];
        local$$27191[_0x34b6[220]](local$$27180[local$$27175 + 0]);
        local$$27191[_0x34b6[220]](local$$27180[local$$27175 + 1]);
      },
      addFace : function(local$$27211, local$$27212, local$$27213, local$$27214, local$$27215, local$$27216, local$$27217, local$$27218, local$$27219, local$$27220, local$$27221, local$$27222) {
        var local$$27230 = this[_0x34b6[1125]][_0x34b6[223]];
        var local$$27236 = this[_0x34b6[1131]](local$$27211, local$$27230);
        var local$$27242 = this[_0x34b6[1131]](local$$27212, local$$27230);
        var local$$27248 = this[_0x34b6[1131]](local$$27213, local$$27230);
        var local$$27250;
        if (local$$27214 === undefined) {
          this[_0x34b6[1132]](local$$27236, local$$27242, local$$27248);
        } else {
          local$$27250 = this[_0x34b6[1131]](local$$27214, local$$27230);
          this[_0x34b6[1132]](local$$27236, local$$27242, local$$27250);
          this[_0x34b6[1132]](local$$27242, local$$27248, local$$27250);
        }
        if (local$$27215 !== undefined) {
          var local$$27285 = this[_0x34b6[1130]][_0x34b6[223]];
          local$$27236 = this[_0x34b6[1133]](local$$27215, local$$27285);
          local$$27242 = this[_0x34b6[1133]](local$$27216, local$$27285);
          local$$27248 = this[_0x34b6[1133]](local$$27217, local$$27285);
          if (local$$27214 === undefined) {
            this[_0x34b6[1134]](local$$27236, local$$27242, local$$27248);
          } else {
            local$$27250 = this[_0x34b6[1133]](local$$27218, local$$27285);
            this[_0x34b6[1134]](local$$27236, local$$27242, local$$27250);
            this[_0x34b6[1134]](local$$27242, local$$27248, local$$27250);
          }
        }
        if (local$$27219 !== undefined) {
          var local$$27340 = this[_0x34b6[1129]][_0x34b6[223]];
          local$$27236 = this[_0x34b6[1135]](local$$27219, local$$27340);
          local$$27242 = local$$27219 === local$$27220 ? local$$27236 : this[_0x34b6[1135]](local$$27220, local$$27340);
          local$$27248 = local$$27219 === local$$27221 ? local$$27236 : this[_0x34b6[1135]](local$$27221, local$$27340);
          if (local$$27214 === undefined) {
            this[_0x34b6[1136]](local$$27236, local$$27242, local$$27248);
          } else {
            local$$27250 = this[_0x34b6[1135]](local$$27222, local$$27340);
            this[_0x34b6[1136]](local$$27236, local$$27242, local$$27250);
            this[_0x34b6[1136]](local$$27242, local$$27248, local$$27250);
          }
        }
      },
      addLineGeometry : function(local$$27393, local$$27394) {
        this[_0x34b6[368]][_0x34b6[1126]][_0x34b6[445]] = _0x34b6[1137];
        var local$$27415 = this[_0x34b6[1125]][_0x34b6[223]];
        var local$$27423 = this[_0x34b6[1130]][_0x34b6[223]];
        /** @type {number} */
        var local$$27426 = 0;
        var local$$27431 = local$$27393[_0x34b6[223]];
        for (; local$$27426 < local$$27431; local$$27426++) {
          this[_0x34b6[1138]](this[_0x34b6[1131]](local$$27393[local$$27426], local$$27415));
        }
        /** @type {number} */
        var local$$27450 = 0;
        local$$27431 = local$$27394[_0x34b6[223]];
        for (; local$$27450 < local$$27431; local$$27450++) {
          this[_0x34b6[1139]](this[_0x34b6[1133]](local$$27394[local$$27450], local$$27423));
        }
      }
    };
    local$$27475[_0x34b6[1140]](_0x34b6[381], false);
    return local$$27475;
  },
  parse : function(local$$27489) {
    console[_0x34b6[1141]](_0x34b6[1112]);
    var local$$27500 = this._createParserState();
    if (local$$27489[_0x34b6[742]](_0x34b6[1142]) !== -1) {
      local$$27489 = local$$27489[_0x34b6[626]](/\r\n/g, _0x34b6[1]);
    }
    if (local$$27489[_0x34b6[742]](_0x34b6[1143]) !== -1) {
      local$$27489 = local$$27489[_0x34b6[626]](/\\\n/g, _0x34b6[381]);
    }
    var local$$27550 = local$$27489[_0x34b6[379]](_0x34b6[1]);
    var local$$27554 = _0x34b6[381];
    var local$$27558 = _0x34b6[381];
    var local$$27562 = _0x34b6[381];
    /** @type {number} */
    var local$$27565 = 0;
    /** @type {!Array} */
    var local$$27568 = [];
    /** @type {boolean} */
    var local$$27579 = typeof _0x34b6[381][_0x34b6[1144]] === _0x34b6[391];
    /** @type {number} */
    var local$$27582 = 0;
    var local$$27587 = local$$27550[_0x34b6[223]];
    for (; local$$27582 < local$$27587; local$$27582++) {
      local$$27554 = local$$27550[local$$27582];
      local$$27554 = local$$27579 ? local$$27554[_0x34b6[1144]]() : local$$27554[_0x34b6[712]]();
      local$$27565 = local$$27554[_0x34b6[223]];
      if (local$$27565 === 0) {
        continue;
      }
      local$$27558 = local$$27554[_0x34b6[1066]](0);
      if (local$$27558 === _0x34b6[1067]) {
        continue;
      }
      if (local$$27558 === _0x34b6[1145]) {
        local$$27562 = local$$27554[_0x34b6[1066]](1);
        if (local$$27562 === _0x34b6[711] && (local$$27568 = this[_0x34b6[1113]][_0x34b6[1147]][_0x34b6[1146]](local$$27554)) !== null) {
          local$$27500[_0x34b6[1125]][_0x34b6[220]](parseFloat(local$$27568[1]), parseFloat(local$$27568[2]), parseFloat(local$$27568[3]));
        } else {
          if (local$$27562 === _0x34b6[1148] && (local$$27568 = this[_0x34b6[1113]][_0x34b6[1149]][_0x34b6[1146]](local$$27554)) !== null) {
            local$$27500[_0x34b6[1129]][_0x34b6[220]](parseFloat(local$$27568[1]), parseFloat(local$$27568[2]), parseFloat(local$$27568[3]));
          } else {
            if (local$$27562 === _0x34b6[1150] && (local$$27568 = this[_0x34b6[1113]][_0x34b6[1151]][_0x34b6[1146]](local$$27554)) !== null) {
              local$$27500[_0x34b6[1130]][_0x34b6[220]](parseFloat(local$$27568[1]), parseFloat(local$$27568[2]));
            } else {
              throw new Error(_0x34b6[1152] + local$$27554 + _0x34b6[1051]);
            }
          }
        }
      } else {
        if (local$$27558 === _0x34b6[1153]) {
          if ((local$$27568 = this[_0x34b6[1113]][_0x34b6[1154]][_0x34b6[1146]](local$$27554)) !== null) {
            local$$27500[_0x34b6[1155]](local$$27568[1], local$$27568[4], local$$27568[7], local$$27568[10], local$$27568[2], local$$27568[5], local$$27568[8], local$$27568[11], local$$27568[3], local$$27568[6], local$$27568[9], local$$27568[12]);
          } else {
            if ((local$$27568 = this[_0x34b6[1113]][_0x34b6[1156]][_0x34b6[1146]](local$$27554)) !== null) {
              local$$27500[_0x34b6[1155]](local$$27568[1], local$$27568[3], local$$27568[5], local$$27568[7], local$$27568[2], local$$27568[4], local$$27568[6], local$$27568[8]);
            } else {
              if ((local$$27568 = this[_0x34b6[1113]][_0x34b6[1157]][_0x34b6[1146]](local$$27554)) !== null) {
                local$$27500[_0x34b6[1155]](local$$27568[1], local$$27568[3], local$$27568[5], local$$27568[7], undefined, undefined, undefined, undefined, local$$27568[2], local$$27568[4], local$$27568[6], local$$27568[8]);
              } else {
                if ((local$$27568 = this[_0x34b6[1113]][_0x34b6[1158]][_0x34b6[1146]](local$$27554)) !== null) {
                  local$$27500[_0x34b6[1155]](local$$27568[1], local$$27568[2], local$$27568[3], local$$27568[4]);
                } else {
                  throw new Error(_0x34b6[1159] + local$$27554 + _0x34b6[1051]);
                }
              }
            }
          }
        } else {
          if (local$$27558 === _0x34b6[1160]) {
            var local$$27936 = local$$27554[_0x34b6[474]](1)[_0x34b6[712]]()[_0x34b6[379]](_0x34b6[711]);
            /** @type {!Array} */
            var local$$27939 = [];
            /** @type {!Array} */
            var local$$27942 = [];
            if (local$$27554[_0x34b6[742]](_0x34b6[1161]) === -1) {
              local$$27939 = local$$27936;
            } else {
              /** @type {number} */
              var local$$27956 = 0;
              var local$$27961 = local$$27936[_0x34b6[223]];
              for (; local$$27956 < local$$27961; local$$27956++) {
                var local$$27973 = local$$27936[local$$27956][_0x34b6[379]](_0x34b6[1161]);
                if (local$$27973[0] !== _0x34b6[381]) {
                  local$$27939[_0x34b6[220]](local$$27973[0]);
                }
                if (local$$27973[1] !== _0x34b6[381]) {
                  local$$27942[_0x34b6[220]](local$$27973[1]);
                }
              }
            }
            local$$27500[_0x34b6[1162]](local$$27939, local$$27942);
          } else {
            if ((local$$27568 = this[_0x34b6[1113]][_0x34b6[1163]][_0x34b6[1146]](local$$27554)) !== null) {
              var local$$28047 = (_0x34b6[711] + local$$27568[0][_0x34b6[702]](1)[_0x34b6[712]]())[_0x34b6[702]](1);
              local$$27500[_0x34b6[1140]](local$$28047);
            } else {
              if (this[_0x34b6[1113]][_0x34b6[1164]][_0x34b6[386]](local$$27554)) {
                local$$27500[_0x34b6[368]][_0x34b6[1166]](local$$27554[_0x34b6[474]](7)[_0x34b6[712]](), local$$27500[_0x34b6[1165]]);
              } else {
                if (this[_0x34b6[1113]][_0x34b6[1167]][_0x34b6[386]](local$$27554)) {
                  local$$27500[_0x34b6[1165]][_0x34b6[220]](local$$27554[_0x34b6[474]](7)[_0x34b6[712]]());
                } else {
                  if ((local$$27568 = this[_0x34b6[1113]][_0x34b6[1168]][_0x34b6[1146]](local$$27554)) !== null) {
                    var local$$28137 = local$$27568[1][_0x34b6[712]]()[_0x34b6[387]]();
                    /** @type {boolean} */
                    local$$27500[_0x34b6[368]][_0x34b6[1121]] = local$$28137 === _0x34b6[1169] || local$$28137 === _0x34b6[1170];
                    var local$$28161 = local$$27500[_0x34b6[368]][_0x34b6[1116]]();
                    if (local$$28161) {
                      local$$28161[_0x34b6[1121]] = local$$27500[_0x34b6[368]][_0x34b6[1121]];
                    }
                  } else {
                    if (local$$27554 === _0x34b6[1171]) {
                      continue;
                    }
                    throw new Error(_0x34b6[1172] + local$$27554 + _0x34b6[1051]);
                  }
                }
              }
            }
          }
        }
      }
    }
    local$$27500[_0x34b6[1173]]();
    var local$$28216 = new THREE.Group;
    local$$28216[_0x34b6[1165]] = [][_0x34b6[611]](local$$27500[_0x34b6[1165]]);
    /** @type {number} */
    local$$27582 = 0;
    local$$27587 = local$$27500[_0x34b6[1128]][_0x34b6[223]];
    for (; local$$27582 < local$$27587; local$$27582++) {
      var local$$28249 = local$$27500[_0x34b6[1128]][local$$27582];
      var local$$28254 = local$$28249[_0x34b6[1126]];
      var local$$28259 = local$$28249[_0x34b6[1078]];
      /** @type {boolean} */
      var local$$28267 = local$$28254[_0x34b6[445]] === _0x34b6[1137];
      if (local$$28254[_0x34b6[1125]][_0x34b6[223]] === 0) {
        continue;
      }
      var local$$28283 = new THREE.BufferGeometry;
      local$$28283[_0x34b6[1174]](_0x34b6[430], new THREE.BufferAttribute(new Float32Array(local$$28254[_0x34b6[1125]]), 3));
      if (local$$28254[_0x34b6[1129]][_0x34b6[223]] > 0) {
        local$$28283[_0x34b6[1174]](_0x34b6[570], new THREE.BufferAttribute(new Float32Array(local$$28254[_0x34b6[1129]]), 3));
      } else {
        local$$28283[_0x34b6[1175]]();
      }
      if (local$$28254[_0x34b6[1130]][_0x34b6[223]] > 0) {
        local$$28283[_0x34b6[1174]](_0x34b6[1176], new THREE.BufferAttribute(new Float32Array(local$$28254[_0x34b6[1130]]), 2));
      }
      /** @type {!Array} */
      var local$$28358 = [];
      /** @type {number} */
      var local$$28361 = 0;
      var local$$28366 = local$$28259[_0x34b6[223]];
      for (; local$$28361 < local$$28366; local$$28361++) {
        var local$$28372 = local$$28259[local$$28361];
        local$$28161 = undefined;
        if (this[_0x34b6[1078]] !== null) {
          local$$28161 = this[_0x34b6[1078]][_0x34b6[242]](local$$28372[_0x34b6[1115]]);
          if (local$$28267 && local$$28161 && !(local$$28161 instanceof THREE[_0x34b6[1177]])) {
            var local$$28402 = new THREE.LineBasicMaterial;
            local$$28402[_0x34b6[338]](local$$28161);
            local$$28161 = local$$28402;
          }
        }
        if (!local$$28161) {
          local$$28161 = !local$$28267 ? new THREE.MeshPhongMaterial : new THREE.LineBasicMaterial;
          local$$28161[_0x34b6[1115]] = local$$28372[_0x34b6[1115]];
        }
        local$$28161[_0x34b6[1178]] = local$$28372[_0x34b6[1121]] ? THREE[_0x34b6[1179]] : THREE[_0x34b6[1180]];
        local$$28358[_0x34b6[220]](local$$28161);
      }
      var local$$28459;
      if (local$$28358[_0x34b6[223]] > 1) {
        /** @type {number} */
        local$$28361 = 0;
        local$$28366 = local$$28259[_0x34b6[223]];
        for (; local$$28361 < local$$28366; local$$28361++) {
          local$$28372 = local$$28259[local$$28361];
          local$$28283[_0x34b6[1181]](local$$28372[_0x34b6[1127]], local$$28372[_0x34b6[1119]], local$$28361);
        }
        var local$$28497 = new THREE.MultiMaterial(local$$28358);
        local$$28459 = !local$$28267 ? new THREE.Mesh(local$$28283, local$$28497) : new THREE.LineSegments(local$$28283, local$$28497);
      } else {
        local$$28459 = !local$$28267 ? new THREE.Mesh(local$$28283, local$$28358[0]) : new THREE.LineSegments(local$$28283, local$$28358[0]);
      }
      local$$28459[_0x34b6[1115]] = local$$28249[_0x34b6[1115]];
      local$$28216[_0x34b6[274]](local$$28459);
    }
    console[_0x34b6[1182]](_0x34b6[1112]);
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
  this[_0x34b6[432]] = local$$28555;
  this[_0x34b6[434]] = local$$28556;
  this[_0x34b6[655]] = local$$28557;
  this[_0x34b6[656]] = local$$28558;
}
/** @type {function(?, ?, ?, ?): undefined} */
LSJRectangle[_0x34b6[219]][_0x34b6[1183]] = LSJRectangle;
/**
 * @param {?} local$$28596
 * @return {?}
 */
LSJRectangle[_0x34b6[219]][_0x34b6[1184]] = function(local$$28596) {
  return this[_0x34b6[655]] >= local$$28596[_0x34b6[432]] && this[_0x34b6[432]] <= local$$28596[_0x34b6[655]] && this[_0x34b6[434]] >= local$$28596[_0x34b6[656]] && this[_0x34b6[656]] <= local$$28596[_0x34b6[434]];
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1185]] = function() {
  this[_0x34b6[1186]] = THREE[_0x34b6[1185]][_0x34b6[768]];
};
THREE[_0x34b6[1185]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1187]][_0x34b6[219]]);
THREE[_0x34b6[1185]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1185]];
/**
 * @param {number} local$$28693
 * @param {boolean} local$$28694
 * @return {?}
 */
THREE[_0x34b6[1185]][_0x34b6[768]] = function(local$$28693, local$$28694) {
  /**
   * @param {?} local$$28697
   * @return {?}
   */
  function local$$28696(local$$28697) {
    return local$$28697[_0x34b6[380]](0) + (local$$28697[_0x34b6[380]](1) << 8) + (local$$28697[_0x34b6[380]](2) << 16) + (local$$28697[_0x34b6[380]](3) << 24);
  }
  /**
   * @param {number} local$$28732
   * @return {?}
   */
  function local$$28731(local$$28732) {
    return String[_0x34b6[377]](local$$28732 & 255, local$$28732 >> 8 & 255, local$$28732 >> 16 & 255, local$$28732 >> 24 & 255);
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
  var local$$28928 = local$$28696(_0x34b6[1188]);
  var local$$28933 = local$$28696(_0x34b6[1189]);
  var local$$28938 = local$$28696(_0x34b6[1190]);
  var local$$28943 = local$$28696(_0x34b6[1191]);
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
    console[_0x34b6[217]](_0x34b6[1192]);
    return local$$28845;
  }
  if (!local$$29002[local$$28967] & local$$28914) {
    console[_0x34b6[217]](_0x34b6[1193]);
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
      local$$28845[_0x34b6[1194]] = THREE[_0x34b6[1195]];
      break;
    case local$$28933:
      /** @type {number} */
      local$$29031 = 16;
      local$$28845[_0x34b6[1194]] = THREE[_0x34b6[1196]];
      break;
    case local$$28938:
      /** @type {number} */
      local$$29031 = 16;
      local$$28845[_0x34b6[1194]] = THREE[_0x34b6[1197]];
      break;
    case local$$28943:
      /** @type {number} */
      local$$29031 = 8;
      local$$28845[_0x34b6[1194]] = THREE[_0x34b6[1198]];
      break;
    default:
      if (local$$29002[local$$28973] === 32 && local$$29002[local$$28976] & 16711680 && local$$29002[local$$28979] & 65280 && local$$29002[local$$28982] & 255 && local$$29002[local$$28985] & 4278190080) {
        /** @type {boolean} */
        local$$29037 = true;
        /** @type {number} */
        local$$29031 = 64;
        local$$28845[_0x34b6[1194]] = THREE[_0x34b6[206]];
      } else {
        console[_0x34b6[217]](_0x34b6[1199], local$$28731(local$$29034));
        return local$$28845;
      }
  }
  /** @type {number} */
  local$$28845[_0x34b6[1200]] = 1;
  if (local$$29002[local$$28955] & local$$28866 && local$$28694 !== false) {
    local$$28845[_0x34b6[1200]] = Math[_0x34b6[532]](1, local$$29002[local$$28964]);
  }
  /** @type {number} */
  var local$$29170 = local$$29002[local$$28991];
  /** @type {boolean} */
  local$$28845[_0x34b6[1201]] = local$$29170 & local$$28884 ? true : false;
  if (local$$28845[_0x34b6[1201]] && (!(local$$29170 & local$$28887) || !(local$$29170 & local$$28890) || !(local$$29170 & local$$28893) || !(local$$29170 & local$$28896) || !(local$$29170 & local$$28899) || !(local$$29170 & local$$28902))) {
    console[_0x34b6[217]](_0x34b6[1202]);
    return local$$28845;
  }
  /** @type {number} */
  local$$28845[_0x34b6[208]] = local$$29002[local$$28961];
  /** @type {number} */
  local$$28845[_0x34b6[209]] = local$$29002[local$$28958];
  /** @type {number} */
  var local$$29228 = local$$29002[local$$28952] + 4;
  /** @type {number} */
  var local$$29236 = local$$28845[_0x34b6[1201]] ? 6 : 1;
  /** @type {number} */
  var local$$29239 = 0;
  for (; local$$29239 < local$$29236; local$$29239++) {
    var local$$29247 = local$$28845[_0x34b6[208]];
    var local$$29252 = local$$28845[_0x34b6[209]];
    /** @type {number} */
    var local$$29255 = 0;
    for (; local$$29255 < local$$28845[_0x34b6[1200]]; local$$29255++) {
      if (local$$29037) {
        var local$$29264 = local$$28755(local$$28693, local$$29228, local$$29247, local$$29252);
        var local$$29269 = local$$29264[_0x34b6[223]];
      } else {
        /** @type {number} */
        local$$29269 = Math[_0x34b6[532]](4, local$$29247) / 4 * Math[_0x34b6[532]](4, local$$29252) / 4 * local$$29031;
        /** @type {!Uint8Array} */
        local$$29264 = new Uint8Array(local$$28693, local$$29228, local$$29269);
      }
      var local$$29297 = {
        "data" : local$$29264,
        "width" : local$$29247,
        "height" : local$$29252
      };
      local$$28845[_0x34b6[1203]][_0x34b6[220]](local$$29297);
      local$$29228 = local$$29228 + local$$29269;
      local$$29247 = Math[_0x34b6[532]](local$$29247 >> 1, 1);
      local$$29252 = Math[_0x34b6[532]](local$$29252 >> 1, 1);
    }
  }
  return local$$28845;
};
/**
 * @param {string} local$$29343
 * @return {undefined}
 */
THREE[_0x34b6[1204]] = function(local$$29343) {
  this[_0x34b6[1055]] = local$$29343 !== undefined ? local$$29343 : THREE[_0x34b6[1056]];
  this[_0x34b6[1186]] = THREE[_0x34b6[1204]][_0x34b6[768]];
};
THREE[_0x34b6[1204]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1187]][_0x34b6[219]]);
THREE[_0x34b6[1204]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1204]];
/**
 * @param {number} local$$29408
 * @param {?} local$$29409
 * @return {?}
 */
THREE[_0x34b6[1204]][_0x34b6[768]] = function(local$$29408, local$$29409) {
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
    return THREE[_0x34b6[1204]]._parseV3(local$$29421);
  } else {
    if (local$$29417[11] === 559044176) {
      return THREE[_0x34b6[1204]]._parseV2(local$$29421);
    } else {
      throw new Error(_0x34b6[1205]);
    }
  }
};
/**
 * @param {?} local$$29463
 * @return {?}
 */
THREE[_0x34b6[1204]][_0x34b6[1206]] = function(local$$29463) {
  var local$$29468 = local$$29463[_0x34b6[1207]];
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
      local$$29472 = THREE[_0x34b6[1208]];
      break;
    case 1:
      /** @type {number} */
      local$$29470 = 2;
      local$$29472 = THREE[_0x34b6[1209]];
      break;
    case 2:
      /** @type {number} */
      local$$29470 = 4;
      local$$29472 = THREE[_0x34b6[1210]];
      break;
    case 3:
      /** @type {number} */
      local$$29470 = 4;
      local$$29472 = THREE[_0x34b6[1211]];
      break;
    default:
      throw new Error(_0x34b6[1212] + local$$29480);
  }
  local$$29463[_0x34b6[1213]] = 52 + local$$29476;
  /** @type {number} */
  local$$29463[_0x34b6[1214]] = local$$29470;
  local$$29463[_0x34b6[1194]] = local$$29472;
  local$$29463[_0x34b6[208]] = local$$29488;
  local$$29463[_0x34b6[209]] = local$$29484;
  local$$29463[_0x34b6[1215]] = local$$29496;
  local$$29463[_0x34b6[1216]] = local$$29500;
  /** @type {boolean} */
  local$$29463[_0x34b6[1201]] = local$$29496 === 6;
  return THREE[_0x34b6[1204]]._extract(local$$29463);
};
/**
 * @param {?} local$$29619
 * @return {?}
 */
THREE[_0x34b6[1204]][_0x34b6[1217]] = function(local$$29619) {
  var local$$29624 = local$$29619[_0x34b6[1207]];
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
    local$$29690 = local$$29694 ? THREE[_0x34b6[1211]] : THREE[_0x34b6[1210]];
    /** @type {number} */
    local$$29652 = 4;
  } else {
    if (local$$29688 === local$$29682) {
      local$$29690 = local$$29694 ? THREE[_0x34b6[1209]] : THREE[_0x34b6[1208]];
      /** @type {number} */
      local$$29652 = 2;
    } else {
      throw new Error(_0x34b6[1218] + local$$29688);
    }
  }
  local$$29619[_0x34b6[1213]] = local$$29628;
  /** @type {number} */
  local$$29619[_0x34b6[1214]] = local$$29652;
  local$$29619[_0x34b6[1194]] = local$$29690;
  local$$29619[_0x34b6[208]] = local$$29636;
  local$$29619[_0x34b6[209]] = local$$29632;
  local$$29619[_0x34b6[1215]] = local$$29676;
  local$$29619[_0x34b6[1216]] = local$$29640 + 1;
  /** @type {boolean} */
  local$$29619[_0x34b6[1201]] = local$$29676 === 6;
  return THREE[_0x34b6[1204]]._extract(local$$29619);
};
/**
 * @param {?} local$$29794
 * @return {?}
 */
THREE[_0x34b6[1204]][_0x34b6[1219]] = function(local$$29794) {
  var local$$29813 = {
    mipmaps : [],
    width : local$$29794[_0x34b6[208]],
    height : local$$29794[_0x34b6[209]],
    format : local$$29794[_0x34b6[1194]],
    mipmapCount : local$$29794[_0x34b6[1216]],
    isCubemap : local$$29794[_0x34b6[1201]]
  };
  var local$$29818 = local$$29794[_0x34b6[1220]];
  var local$$29823 = local$$29794[_0x34b6[1213]];
  var local$$29828 = local$$29794[_0x34b6[1214]];
  var local$$29833 = local$$29794[_0x34b6[1215]];
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
  local$$29813[_0x34b6[1203]][_0x34b6[223]] = local$$29794[_0x34b6[1216]] * local$$29833;
  /** @type {number} */
  var local$$29890 = 0;
  for (; local$$29890 < local$$29794[_0x34b6[1216]];) {
    /** @type {number} */
    var local$$29902 = local$$29794[_0x34b6[208]] >> local$$29890;
    /** @type {number} */
    var local$$29908 = local$$29794[_0x34b6[209]] >> local$$29890;
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
      local$$29813[_0x34b6[1203]][local$$29937 * local$$29794[_0x34b6[1216]] + local$$29890] = local$$29946;
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
  var local$$29986 = window[_0x34b6[1222]][_0x34b6[1221]];
  var local$$29994 = window[_0x34b6[1222]][_0x34b6[1223]];
  var local$$30002 = window[_0x34b6[1222]][_0x34b6[1224]];
  var local$$30010 = window[_0x34b6[1222]][_0x34b6[1225]];
  var local$$30018 = window[_0x34b6[1222]][_0x34b6[1226]];
  var local$$30029 = window[_0x34b6[441]][_0x34b6[603]][_0x34b6[549]];
  var local$$30051 = local$$29986 + _0x34b6[1067] + local$$29994 + _0x34b6[1067] + local$$30002 + _0x34b6[1067] + local$$30010 + _0x34b6[1067] + local$$30018 + _0x34b6[1067] + local$$30029;
  var local$$30063 = _0x34b6[1227] + local$$29978 + _0x34b6[1228] + local$$30029 + _0x34b6[1229];
  /** @type {boolean} */
  this[_0x34b6[1230]] = false;
  var local$$30071 = this;
  try {
    /** @type {!XMLHttpRequest} */
    var local$$30074 = new XMLHttpRequest;
    local$$30074[_0x34b6[452]](_0x34b6[1231], _0x34b6[1232], false);
    local$$30074[_0x34b6[1235]](_0x34b6[1233], _0x34b6[1234]);
    /**
     * @return {?}
     */
    local$$30074[_0x34b6[1236]] = function() {
      /** @type {!XMLHttpRequest} */
      var local$$30099 = local$$30074;
      if (local$$30099[_0x34b6[1237]] == 4) {
        if (local$$30099[_0x34b6[1045]] == 200) {
          var local$$30114 = local$$30099[_0x34b6[1046]];
          if (local$$30114[_0x34b6[712]]() == _0x34b6[1169]) {
            /** @type {boolean} */
            local$$30071[_0x34b6[1230]] = true;
            return true;
          }
        } else {
          /** @type {boolean} */
          local$$30071[_0x34b6[1230]] = false;
          return false;
        }
      }
    };
    local$$30074[_0x34b6[1049]](local$$30063);
  } catch (local$$30156) {
  }
  return this[_0x34b6[1230]];
}
/**
 * @param {?} local$$30169
 * @return {?}
 */
function checkPrivateLicense(local$$30169) {
  /** @type {boolean} */
  this[_0x34b6[1230]] = false;
  var local$$30177 = this;
  try {
    /** @type {!XMLHttpRequest} */
    var local$$30180 = new XMLHttpRequest;
    local$$30180[_0x34b6[452]](_0x34b6[1231], _0x34b6[1238], false);
    local$$30180[_0x34b6[1235]](_0x34b6[1233], _0x34b6[1234]);
    /**
     * @return {?}
     */
    local$$30180[_0x34b6[1236]] = function() {
      /** @type {!XMLHttpRequest} */
      var local$$30205 = local$$30180;
      if (local$$30205[_0x34b6[1237]] == 4) {
        if (local$$30205[_0x34b6[1045]] == 200) {
          var local$$30220 = local$$30205[_0x34b6[1046]];
          if (local$$30220[_0x34b6[712]]() == local$$30169) {
            /** @type {boolean} */
            local$$30177[_0x34b6[1230]] = true;
            return true;
          } else {
            /** @type {boolean} */
            local$$30177[_0x34b6[1230]] = false;
            return false;
          }
        } else {
          /** @type {boolean} */
          local$$30177[_0x34b6[1230]] = false;
          return false;
        }
      }
    };
    local$$30180[_0x34b6[1049]]();
  } catch (local$$30269) {
  }
  return this[_0x34b6[1230]];
}
/**
 * @return {undefined}
 */
LSJCamera = function() {
  this[_0x34b6[1239]] = new THREE.Quaternion;
  this[_0x34b6[430]] = new THREE.Vector3;
};
/** @type {function(): undefined} */
LSJCamera[_0x34b6[219]][_0x34b6[1183]] = LSJCamera;
/**
 * @return {?}
 */
LSJCamera[_0x34b6[219]][_0x34b6[1240]] = function() {
  return this[_0x34b6[430]];
};
/**
 * @param {?} local$$30331
 * @return {undefined}
 */
LSJCamera[_0x34b6[219]][_0x34b6[1241]] = function(local$$30331) {
  this[_0x34b6[430]][_0x34b6[338]](local$$30331);
};
/**
 * @return {?}
 */
LSJCamera[_0x34b6[219]][_0x34b6[1242]] = function() {
  return this[_0x34b6[1239]];
};
/**
 * @param {?} local$$30366
 * @return {undefined}
 */
LSJCamera[_0x34b6[219]][_0x34b6[1243]] = function(local$$30366) {
  this[_0x34b6[1239]][_0x34b6[338]](local$$30366);
};
/**
 * @param {?} local$$30383
 * @return {undefined}
 */
LSJCameraTrackControls = function(local$$30383) {
  this[_0x34b6[240]] = local$$30383;
  /** @type {boolean} */
  this[_0x34b6[359]] = false;
  /** @type {!Array} */
  this[_0x34b6[1244]] = [];
  /** @type {number} */
  this[_0x34b6[1245]] = 0;
  var local$$30408 = this;
  this[_0x34b6[1246]] = undefined;
  this[_0x34b6[1247]] = undefined;
  this[_0x34b6[1248]] = undefined;
  this[_0x34b6[1249]] = undefined;
  /** @type {boolean} */
  this[_0x34b6[1250]] = false;
  this[_0x34b6[1251]] = Date[_0x34b6[348]]();
  this[_0x34b6[1252]] = Date[_0x34b6[348]]();
  /**
   * @return {undefined}
   */
  this[_0x34b6[1253]] = function() {
    /** @type {!Array} */
    var local$$30459 = [];
    /** @type {!Array} */
    var local$$30462 = [];
    /** @type {!Array} */
    var local$$30465 = [];
    /** @type {number} */
    var local$$30468 = 0;
    for (; local$$30468 < this[_0x34b6[1244]][_0x34b6[223]]; local$$30468++) {
      var local$$30483 = this[_0x34b6[1244]][local$$30468];
      local$$30459[_0x34b6[220]](local$$30483[_0x34b6[1141]]);
      local$$30483[_0x34b6[275]][_0x34b6[1240]]()[_0x34b6[1254]](local$$30462, local$$30462[_0x34b6[223]]);
      local$$30483[_0x34b6[275]][_0x34b6[1242]]()[_0x34b6[1254]](local$$30465, local$$30465[_0x34b6[223]]);
      this[_0x34b6[1245]] = local$$30483[_0x34b6[1141]];
    }
    this[_0x34b6[1246]] = new THREE.VectorKeyframeTrack(_0x34b6[1255], local$$30459, local$$30462);
    this[_0x34b6[1248]] = this[_0x34b6[1246]][_0x34b6[1256]](undefined);
    this[_0x34b6[1247]] = new THREE.QuaternionKeyframeTrack(_0x34b6[1257], local$$30459, local$$30465);
    this[_0x34b6[1249]] = this[_0x34b6[1247]][_0x34b6[1256]](undefined);
  };
  /**
   * @return {?}
   */
  this[_0x34b6[1258]] = function() {
    return this[_0x34b6[1244]];
  };
  /**
   * @return {?}
   */
  this[_0x34b6[235]] = function() {
    return this[_0x34b6[1244]] = [];
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1259]] = function() {
    if (this[_0x34b6[359]]) {
      return;
    }
    if (!this[_0x34b6[1250]]) {
      this[_0x34b6[1253]]();
      /** @type {boolean} */
      this[_0x34b6[359]] = true;
      this[_0x34b6[1251]] = Date[_0x34b6[348]]();
    } else {
      /** @type {boolean} */
      this[_0x34b6[359]] = true;
      /** @type {number} */
      this[_0x34b6[1251]] = Date[_0x34b6[348]]() - (this[_0x34b6[1252]] - this[_0x34b6[1251]]);
      /** @type {boolean} */
      this[_0x34b6[1250]] = false;
    }
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[661]] = function() {
    /** @type {boolean} */
    this[_0x34b6[359]] = false;
    /** @type {number} */
    this[_0x34b6[1251]] = 0;
    /** @type {boolean} */
    this[_0x34b6[1250]] = false;
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1260]] = function() {
    if (!this[_0x34b6[1250]]) {
      /** @type {boolean} */
      this[_0x34b6[359]] = false;
      /** @type {boolean} */
      this[_0x34b6[1250]] = true;
      this[_0x34b6[1252]] = Date[_0x34b6[348]]();
    }
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1261]] = function() {
    if (!this[_0x34b6[359]]) {
      return;
    }
    var local$$30756 = Date[_0x34b6[348]]();
    if (local$$30756 - this[_0x34b6[1251]] > this[_0x34b6[1245]]) {
      /** @type {boolean} */
      this[_0x34b6[359]] = false;
      var local$$30784 = this[_0x34b6[1244]][this[_0x34b6[1244]][_0x34b6[223]] - 1];
      local$$30408[_0x34b6[240]][_0x34b6[430]][_0x34b6[338]](local$$30784[_0x34b6[430]]);
      local$$30408[_0x34b6[240]][_0x34b6[1239]][_0x34b6[338]](local$$30784[_0x34b6[1239]]);
      return;
    }
    var local$$30829 = this[_0x34b6[1248]][_0x34b6[1262]](local$$30756 - this[_0x34b6[1251]]);
    local$$30408[_0x34b6[240]][_0x34b6[430]][_0x34b6[462]](local$$30829);
    var local$$30853 = this[_0x34b6[1249]][_0x34b6[1262]](local$$30756 - this[_0x34b6[1251]]);
    local$$30408[_0x34b6[240]][_0x34b6[1239]][_0x34b6[462]](local$$30853);
    local$$30408[_0x34b6[240]][_0x34b6[1263]]();
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
    local$$30921 = local$$30922[_0x34b6[1264]]();
    local$$30929 = local$$30922[_0x34b6[1264]]();
    local$$30922[_0x34b6[1265]](local$$30922.ARRAY_BUFFER, local$$30921);
    local$$30922[_0x34b6[1266]](local$$30922.ARRAY_BUFFER, local$$30908, local$$30922.STATIC_DRAW);
    local$$30922[_0x34b6[1265]](local$$30922.ELEMENT_ARRAY_BUFFER, local$$30929);
    local$$30922[_0x34b6[1266]](local$$30922.ELEMENT_ARRAY_BUFFER, local$$30919, local$$30922.STATIC_DRAW);
    local$$30965 = local$$30966();
    local$$30970 = {
      position : local$$30922[_0x34b6[1267]](local$$30965, _0x34b6[430]),
      uv : local$$30922[_0x34b6[1267]](local$$30965, _0x34b6[1176])
    };
    local$$30986 = {
      uvOffset : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[1268]),
      uvScale : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[1270]),
      rotation : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[1271]),
      scale : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[1090]),
      color : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[245]),
      map : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[645]),
      opacity : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[322]),
      modelViewMatrix : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[1272]),
      projectionMatrix : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[335]),
      alphaTest : local$$30922[_0x34b6[1269]](local$$30965, _0x34b6[1273])
    };
    var local$$31056 = document[_0x34b6[424]](_0x34b6[516]);
    /** @type {number} */
    local$$31056[_0x34b6[208]] = 8;
    /** @type {number} */
    local$$31056[_0x34b6[209]] = 8;
    var local$$31076 = local$$31056[_0x34b6[403]](_0x34b6[402]);
    local$$31076[_0x34b6[976]] = _0x34b6[1274];
    local$$31076[_0x34b6[977]](0, 0, 8, 8);
    local$$31094 = new THREE.Texture(local$$31056);
    /** @type {boolean} */
    local$$31094[_0x34b6[1275]] = true;
  }
  /**
   * @return {?}
   */
  function local$$30966() {
    var local$$31113 = local$$30922[_0x34b6[1314]]();
    var local$$31121 = local$$30922[_0x34b6[1315]](local$$30922.VERTEX_SHADER);
    var local$$31129 = local$$30922[_0x34b6[1315]](local$$30922.FRAGMENT_SHADER);
    local$$30922[_0x34b6[1338]](local$$31121, [_0x34b6[1316] + local$$30885[_0x34b6[1317]]() + _0x34b6[1318], _0x34b6[1319], _0x34b6[1320], _0x34b6[1321], _0x34b6[1322], _0x34b6[1323], _0x34b6[1324], _0x34b6[1325], _0x34b6[1326], _0x34b6[1327], _0x34b6[4], _0x34b6[1328], _0x34b6[1329], _0x34b6[1330], _0x34b6[1331], _0x34b6[1332], _0x34b6[1333], _0x34b6[1334], _0x34b6[1335], _0x34b6[1336], _0x34b6[1337], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]));
    local$$30922[_0x34b6[1338]](local$$31129, [_0x34b6[1316] + local$$30885[_0x34b6[1317]]() + _0x34b6[1318], _0x34b6[1339], _0x34b6[1340], _0x34b6[200], _0x34b6[1341], _0x34b6[1327], _0x34b6[4], _0x34b6[1342], _0x34b6[1343], _0x34b6[1344], _0x34b6[7]][_0x34b6[2]](_0x34b6[1]));
    local$$30922[_0x34b6[1345]](local$$31121);
    local$$30922[_0x34b6[1345]](local$$31129);
    local$$30922[_0x34b6[1346]](local$$31113, local$$31121);
    local$$30922[_0x34b6[1346]](local$$31113, local$$31129);
    local$$30922[_0x34b6[1347]](local$$31113);
    return local$$31113;
  }
  /**
   * @param {?} local$$31266
   * @param {?} local$$31267
   * @return {?}
   */
  function local$$31265(local$$31266, local$$31267) {
    if (local$$31266[_0x34b6[1287]] !== local$$31267[_0x34b6[1287]]) {
      return local$$31267[_0x34b6[1287]] - local$$31266[_0x34b6[1287]];
    } else {
      return local$$31267[_0x34b6[332]] - local$$31266[_0x34b6[332]];
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
    var local$$31307 = local$$31298[_0x34b6[223]];
    for (; local$$31302 < local$$31307; local$$31302++) {
      var local$$31313 = local$$31298[local$$31302];
      var local$$31319 = local$$31313[_0x34b6[1184]](local$$31299);
      if (local$$31319) {
        return true;
      }
    }
    return false;
  }
  var local$$30922 = local$$30885[_0x34b6[227]];
  var local$$31339 = local$$30885[_0x34b6[243]];
  this[_0x34b6[1128]] = local$$30885[_0x34b6[1128]];
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
  this[_0x34b6[225]] = function(local$$31372, local$$31373, local$$31374) {
    local$$30886 = local$$31374[_0x34b6[1276]];
    if (local$$30886[_0x34b6[223]] === 0) {
      return;
    }
    if (local$$30965 === undefined) {
      local$$30888();
    }
    local$$30922[_0x34b6[1277]](local$$30965);
    local$$31339[_0x34b6[1278]]();
    local$$31339[_0x34b6[1279]]();
    local$$31339[_0x34b6[339]](local$$30922.CULL_FACE);
    local$$31339[_0x34b6[359]](local$$30922.BLEND);
    local$$30922[_0x34b6[1265]](local$$30922.ELEMENT_ARRAY_BUFFER, local$$30929);
    local$$30922[_0x34b6[1281]](local$$30986[_0x34b6[335]], false, local$$31373[_0x34b6[335]][_0x34b6[1280]]);
    local$$31339[_0x34b6[1282]](local$$30922.TEXTURE0);
    local$$30922[_0x34b6[1283]](local$$30986[_0x34b6[645]], 0);
    local$$30922[_0x34b6[1283]](local$$30986[_0x34b6[1284]], 0);
    /** @type {number} */
    var local$$31472 = 0;
    var local$$31477 = local$$30886[_0x34b6[223]];
    for (; local$$31472 < local$$31477; local$$31472++) {
      var local$$31483 = local$$30886[local$$31472];
      this[_0x34b6[1128]][_0x34b6[1261]](local$$31483);
      local$$31483[_0x34b6[1272]][_0x34b6[1286]](local$$31373[_0x34b6[337]], local$$31483[_0x34b6[1285]]);
      /** @type {number} */
      local$$31483[_0x34b6[1287]] = -local$$31483[_0x34b6[1272]][_0x34b6[1280]][14];
    }
    local$$30886[_0x34b6[798]](local$$31265);
    /** @type {!Array} */
    var local$$31530 = [];
    /** @type {!Array} */
    var local$$31533 = [];
    /** @type {number} */
    local$$31472 = local$$30886[_0x34b6[223]] - 1;
    for (; local$$31472 >= 0; local$$31472--) {
      local$$31483 = local$$30886[local$$31472];
      var local$$31553 = local$$31483[_0x34b6[1288]]();
      if (local$$31483[_0x34b6[1289]][_0x34b6[445]] != _0x34b6[1290]) {
        if (local$$31297(local$$31533, local$$31553)) {
          if (local$$31483[_0x34b6[1289]][_0x34b6[428]][_0x34b6[1291]] == undefined || local$$31483[_0x34b6[1289]][_0x34b6[428]][_0x34b6[1291]] == _0x34b6[381]) {
            continue;
          }
          local$$31483[_0x34b6[1289]][_0x34b6[1292]](false);
        } else {
          local$$31483[_0x34b6[1289]][_0x34b6[1292]](true);
        }
        local$$31533[_0x34b6[220]](local$$31553);
      }
      var local$$31628 = local$$31483[_0x34b6[1126]][_0x34b6[1293]];
      var local$$31630;
      for (local$$31630 in local$$30970) {
        var local$$31633 = local$$30970[local$$31630];
        if (local$$31633 >= 0) {
          var local$$31638 = local$$31628[local$$31630];
          if (local$$31638 !== undefined) {
            var local$$31644 = local$$31638[_0x34b6[1294]];
            var local$$31653 = this[_0x34b6[1128]][_0x34b6[1295]](local$$31638);
            local$$31339[_0x34b6[1296]](local$$31633);
            local$$30922[_0x34b6[1265]](local$$30922.ARRAY_BUFFER, local$$31653);
            local$$30922[_0x34b6[1297]](local$$31633, local$$31644, local$$30922.FLOAT, false, 0, 0);
          }
        }
      }
      var local$$31686 = local$$31483[_0x34b6[268]];
      local$$30922[_0x34b6[1298]](local$$30986[_0x34b6[1273]], local$$31686[_0x34b6[1273]]);
      local$$30922[_0x34b6[1281]](local$$30986[_0x34b6[1272]], false, local$$31483[_0x34b6[1272]][_0x34b6[1280]]);
      local$$31483[_0x34b6[1285]][_0x34b6[1299]](local$$31359, local$$31363, local$$31367);
      local$$31530[0] = local$$31367[_0x34b6[290]];
      local$$31530[1] = local$$31367[_0x34b6[291]];
      if (local$$31686[_0x34b6[645]] !== null) {
        local$$30922[_0x34b6[1300]](local$$30986[_0x34b6[1268]], local$$31686[_0x34b6[645]][_0x34b6[1091]][_0x34b6[290]], local$$31686[_0x34b6[645]][_0x34b6[1091]][_0x34b6[291]]);
        local$$30922[_0x34b6[1300]](local$$30986[_0x34b6[1270]], local$$31686[_0x34b6[645]][_0x34b6[999]][_0x34b6[290]], local$$31686[_0x34b6[645]][_0x34b6[999]][_0x34b6[291]]);
      } else {
        local$$30922[_0x34b6[1300]](local$$30986[_0x34b6[1268]], 0, 0);
        local$$30922[_0x34b6[1300]](local$$30986[_0x34b6[1270]], 1, 1);
      }
      local$$30922[_0x34b6[1298]](local$$30986[_0x34b6[322]], local$$31686[_0x34b6[322]]);
      local$$30922[_0x34b6[1301]](local$$30986[_0x34b6[245]], local$$31686[_0x34b6[245]][_0x34b6[458]], local$$31686[_0x34b6[245]][_0x34b6[459]], local$$31686[_0x34b6[245]][_0x34b6[460]]);
      local$$30922[_0x34b6[1298]](local$$30986[_0x34b6[1271]], local$$31686[_0x34b6[1271]]);
      local$$30922[_0x34b6[1302]](local$$30986[_0x34b6[1090]], local$$31530);
      local$$31339[_0x34b6[1306]](local$$31686[_0x34b6[301]], local$$31686[_0x34b6[1303]], local$$31686[_0x34b6[1304]], local$$31686[_0x34b6[1305]]);
      local$$31339[_0x34b6[1308]](local$$31686[_0x34b6[1307]]);
      local$$31339[_0x34b6[1310]](local$$31686[_0x34b6[1309]]);
      if (local$$31686[_0x34b6[645]] && local$$31686[_0x34b6[645]][_0x34b6[554]] && local$$31686[_0x34b6[645]][_0x34b6[554]][_0x34b6[208]]) {
        local$$30885[_0x34b6[1311]](local$$31686[_0x34b6[645]], 0);
      } else {
        local$$30885[_0x34b6[1311]](local$$31094, 0);
      }
      local$$30922[_0x34b6[1312]](local$$30922.TRIANGLES, 6, local$$30922.UNSIGNED_SHORT, 0);
    }
    local$$31339[_0x34b6[359]](local$$30922.CULL_FACE);
    local$$30885[_0x34b6[1313]]();
  };
};
/**
 * @param {string} local$$31981
 * @param {?} local$$31982
 * @param {string} local$$31983
 * @return {undefined}
 */
LSJBillboard = function(local$$31981, local$$31982, local$$31983) {
  THREE[_0x34b6[1348]][_0x34b6[238]](this);
  /** @type {!Uint16Array} */
  var local$$32001 = new Uint16Array([0, 1, 2, 0, 2, 3]);
  /** @type {number} */
  var local$$32004 = 1;
  /** @type {number} */
  var local$$32026 = local$$32004 * local$$31981[_0x34b6[645]][_0x34b6[554]][_0x34b6[208]] / local$$31981[_0x34b6[645]][_0x34b6[554]][_0x34b6[209]];
  if (local$$31983 != undefined) {
    var local$$32038 = local$$31983[_0x34b6[1349]] != undefined ? local$$31983[_0x34b6[1349]] : 0;
    /** @type {number} */
    var local$$32050 = local$$31983[_0x34b6[1350]] != undefined ? local$$31983[_0x34b6[1350]] * local$$32026 : 0;
    /** @type {!Float32Array} */
    var local$$32066 = new Float32Array([-local$$32050, -local$$32038, 0, local$$32026 - local$$32050, -local$$32038, 0, local$$32026 - local$$32050, local$$32004 - local$$32038, 0, -local$$32050, local$$32004 - local$$32038, 0]);
    this[_0x34b6[1351]] = new THREE.Vector3(-local$$32050, -local$$32038, 0);
    this[_0x34b6[1352]] = new THREE.Vector3(local$$32026 - local$$32050, -local$$32038, 0);
    this[_0x34b6[1353]] = new THREE.Vector3(local$$32026 - local$$32050, local$$32004 - local$$32038, 0);
    this[_0x34b6[1354]] = new THREE.Vector3(-local$$32050, local$$32004 - local$$32038, 0);
  } else {
    /** @type {!Float32Array} */
    local$$32066 = new Float32Array([-.5, 0, 0, local$$32026 - .5, 0, 0, local$$32026 - .5, local$$32004, 0, -.5, local$$32004, 0]);
    this[_0x34b6[1351]] = new THREE.Vector3(-.5, 0, 0);
    this[_0x34b6[1352]] = new THREE.Vector3(local$$32026, 0, 0);
    this[_0x34b6[1353]] = new THREE.Vector3(local$$32026, local$$32004, 0);
    this[_0x34b6[1354]] = new THREE.Vector3(-.5, local$$32004, 0);
  }
  /** @type {!Float32Array} */
  var local$$32174 = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  var local$$32178 = new THREE.BufferGeometry;
  local$$32178[_0x34b6[1355]](new THREE.BufferAttribute(local$$32001, 1));
  local$$32178[_0x34b6[1174]](_0x34b6[430], new THREE.BufferAttribute(local$$32066, 3));
  local$$32178[_0x34b6[1174]](_0x34b6[1176], new THREE.BufferAttribute(local$$32174, 2));
  this[_0x34b6[445]] = _0x34b6[1356];
  this[_0x34b6[1126]] = local$$32178;
  this[_0x34b6[268]] = local$$31981 !== undefined ? local$$31981 : new THREE.SpriteMaterial({
    depthTest : false
  });
  this[_0x34b6[1289]] = local$$31982;
  this[_0x34b6[240]] = undefined;
};
LSJBillboard[_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1348]][_0x34b6[219]]);
/** @type {function(string, ?, string): undefined} */
LSJBillboard[_0x34b6[219]][_0x34b6[1183]] = LSJBillboard;
LSJBillboard[_0x34b6[219]][_0x34b6[1357]] = function() {
  var local$$32278 = new THREE.Vector3;
  return function local$$32280(local$$32281, local$$32282) {
    if (this[_0x34b6[240]] == undefined) {
      return;
    }
    this[_0x34b6[1263]]();
    var local$$32302 = new THREE.Vector3(0, 0, 0);
    var local$$32306 = new THREE.Matrix4;
    local$$32306[_0x34b6[1286]](this[_0x34b6[240]][_0x34b6[337]], this[_0x34b6[740]]);
    var local$$32326 = local$$32302[_0x34b6[1358]](local$$32306);
    var local$$32334 = this[_0x34b6[240]][_0x34b6[335]];
    var local$$32342 = this[_0x34b6[1090]][_0x34b6[290]];
    var local$$32359 = this[_0x34b6[1351]][_0x34b6[212]]()[_0x34b6[350]](local$$32342)[_0x34b6[274]](local$$32326);
    var local$$32376 = this[_0x34b6[1352]][_0x34b6[212]]()[_0x34b6[350]](local$$32342)[_0x34b6[274]](local$$32326);
    var local$$32393 = this[_0x34b6[1353]][_0x34b6[212]]()[_0x34b6[350]](local$$32342)[_0x34b6[274]](local$$32326);
    var local$$32410 = this[_0x34b6[1354]][_0x34b6[212]]()[_0x34b6[350]](local$$32342)[_0x34b6[274]](local$$32326);
    var local$$32414 = new THREE.Vector3;
    var local$$32418 = new THREE.Matrix4;
    local$$32359 = local$$32359[_0x34b6[1358]](this[_0x34b6[240]][_0x34b6[1285]]);
    local$$32376 = local$$32376[_0x34b6[1358]](this[_0x34b6[240]][_0x34b6[1285]]);
    local$$32393 = local$$32393[_0x34b6[1358]](this[_0x34b6[240]][_0x34b6[1285]]);
    local$$32410 = local$$32410[_0x34b6[1358]](this[_0x34b6[240]][_0x34b6[1285]]);
    if (local$$32281[_0x34b6[1360]][_0x34b6[1359]](local$$32359, local$$32376, local$$32393, false, local$$32414) != null) {
      local$$32282[_0x34b6[220]]({
        distance : local$$32414[_0x34b6[223]](),
        point : local$$32414,
        object : this[_0x34b6[1289]]
      });
    }
    if (local$$32281[_0x34b6[1360]][_0x34b6[1359]](local$$32359, local$$32376, local$$32410, false, local$$32414) != null) {
      local$$32282[_0x34b6[220]]({
        distance : local$$32414[_0x34b6[223]](),
        point : local$$32414,
        object : this[_0x34b6[1289]]
      });
    }
  };
}();
/**
 * @return {?}
 */
LSJBillboard[_0x34b6[219]][_0x34b6[1288]] = function() {
  var local$$32542 = this[_0x34b6[1289]][_0x34b6[1288]]();
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
LSJUtility[_0x34b6[219]][_0x34b6[1183]] = LSJUtility;
/**
 * @param {?} local$$32613
 * @return {?}
 */
LSJUtility[_0x34b6[1361]] = function(local$$32613) {
  if (!LSJMath[_0x34b6[1362]](local$$32613[_0x34b6[208]]) || !LSJMath[_0x34b6[1362]](local$$32613[_0x34b6[209]])) {
    var local$$32639 = document[_0x34b6[424]](_0x34b6[516]);
    local$$32639[_0x34b6[208]] = LSJMath.Fs(local$$32613[_0x34b6[208]]);
    local$$32639[_0x34b6[209]] = LSJMath.Fs(local$$32613[_0x34b6[209]]);
    local$$32639[_0x34b6[403]](_0x34b6[402])[_0x34b6[542]](local$$32613, 0, 0, local$$32613[_0x34b6[208]], local$$32613[_0x34b6[209]], 0, 0, local$$32639[_0x34b6[208]], local$$32639[_0x34b6[209]]);
    return local$$32639;
  }
  return local$$32613;
};
/**
 * @param {?} local$$32700
 * @return {?}
 */
LSJUtility[_0x34b6[1363]] = function(local$$32700) {
  if (!THREE[_0x34b6[1365]][_0x34b6[1364]](local$$32700[_0x34b6[208]]) || !THREE[_0x34b6[1365]][_0x34b6[1364]](local$$32700[_0x34b6[209]])) {
    var local$$32731 = document[_0x34b6[424]](_0x34b6[516]);
    local$$32731[_0x34b6[208]] = LSJMath[_0x34b6[1366]](local$$32700[_0x34b6[208]]);
    local$$32731[_0x34b6[209]] = LSJMath[_0x34b6[1366]](local$$32700[_0x34b6[209]]);
    var local$$32763 = local$$32731[_0x34b6[403]](_0x34b6[402]);
    local$$32763[_0x34b6[542]](local$$32700, 0, 0, local$$32700[_0x34b6[208]], local$$32700[_0x34b6[209]], 0, 0, local$$32731[_0x34b6[208]], local$$32731[_0x34b6[209]]);
    return local$$32731;
  }
  return local$$32700;
};
/**
 * @param {?} local$$32798
 * @param {?} local$$32799
 * @return {?}
 */
LSJUtility[_0x34b6[1367]] = function(local$$32798, local$$32799) {
  var local$$32806 = local$$32798[_0x34b6[702]](0);
  local$$32806[_0x34b6[712]]();
  local$$32806[_0x34b6[626]](_0x34b6[1368], _0x34b6[1161]);
  if (local$$32806[_0x34b6[223]] >= 2) {
    if (local$$32806[_0x34b6[702]](0, 2) == _0x34b6[1369]) {
      local$$32806[0] = _0x34b6[1368];
      local$$32806[1] = _0x34b6[1368];
    }
    local$$32806[_0x34b6[626]](_0x34b6[1369], _0x34b6[1161]);
    if (local$$32806[_0x34b6[223]] > 5) {
      if (local$$32806[_0x34b6[702]](0, 5) == _0x34b6[1370]) {
        /** @type {number} */
        var local$$32875 = 5;
        for (; local$$32875 < local$$32806[_0x34b6[223]]; local$$32875++) {
          if (local$$32806[_0x34b6[1066]](local$$32875) != _0x34b6[1161]) {
            break;
          }
        }
        var local$$32904 = local$$32806[_0x34b6[702]](local$$32875, local$$32806[_0x34b6[223]] - local$$32875);
        local$$32806 = _0x34b6[1371] + local$$32904;
      }
    }
  }
  if (!local$$32799) {
    /** @type {number} */
    var local$$32924 = local$$32806[_0x34b6[223]] - 1;
    for (; local$$32924 >= 0;) {
      if (local$$32806[_0x34b6[1066]](local$$32924) != _0x34b6[1161]) {
        break;
      }
      local$$32924--;
    }
    local$$32806 = local$$32806[_0x34b6[702]](0, local$$32924 + 1);
    local$$32806 = local$$32806 + _0x34b6[1161];
  }
  return local$$32806;
};
/**
 * @param {?} local$$32971
 * @return {?}
 */
LSJUtility[_0x34b6[1372]] = function(local$$32971) {
  var local$$32976 = local$$32971[_0x34b6[223]];
  var local$$32984 = local$$32971[_0x34b6[382]](_0x34b6[1161]);
  if (local$$32984 < 0) {
    local$$32984 = local$$32971[_0x34b6[382]](_0x34b6[1368]);
  }
  if (local$$32984 < 0) {
    return local$$32971[_0x34b6[702]](0);
  }
  var local$$33016 = local$$32971[_0x34b6[702]](0, local$$32984);
  if (local$$33016[_0x34b6[1066]](local$$33016[_0x34b6[223]] - 1) != _0x34b6[1161]) {
    local$$33016 = local$$33016 + _0x34b6[1161];
  }
  return local$$33016;
};
/**
 * @param {?} local$$33048
 * @param {?} local$$33049
 * @return {?}
 */
LSJUtility[_0x34b6[1373]] = function(local$$33048, local$$33049) {
  var local$$33056 = local$$33048[_0x34b6[702]](0);
  var local$$33063 = local$$33049[_0x34b6[702]](0);
  local$$33056[_0x34b6[712]]();
  local$$33063[_0x34b6[712]]();
  var local$$33075;
  if (local$$33056 == _0x34b6[381]) {
    local$$33075 = local$$33063[_0x34b6[702]](0, 2);
    if (local$$33075 == _0x34b6[1374] || local$$33075 == _0x34b6[1375]) {
      return local$$33063[_0x34b6[702]](2);
    }
    local$$33075 = local$$33063[_0x34b6[702]](0, 3);
    if (local$$33075 == _0x34b6[1376] || local$$33075 == _0x34b6[1377]) {
      return _0x34b6[381];
    }
  }
  if (local$$33063 == _0x34b6[381]) {
    return local$$33063;
  }
  if (local$$33063[_0x34b6[223]] >= 2 && local$$33063[_0x34b6[1066]](1) == _0x34b6[1378] || local$$33063[_0x34b6[223]] >= 5 && local$$33063[_0x34b6[1066]](4) == _0x34b6[1378]) {
    return local$$33063;
  }
  local$$33075 = local$$33063[_0x34b6[702]](0, 2);
  if (local$$33075 == _0x34b6[1379] || local$$33075 == _0x34b6[1369]) {
    return local$$33063;
  }
  local$$33056 = LSJUtility[_0x34b6[1367]](local$$33056, false);
  local$$33063 = LSJUtility[_0x34b6[1367]](local$$33063, true);
  local$$33075 = local$$33063[_0x34b6[702]](0, 2);
  var local$$33214 = local$$33063[_0x34b6[702]](0, 3);
  if (local$$33075 == _0x34b6[1374]) {
    local$$33063 = local$$33063[_0x34b6[702]](2);
    return local$$33056 + local$$33063;
  } else {
    if (local$$33063[_0x34b6[223]] >= 2 && local$$33063[_0x34b6[1066]](1) == _0x34b6[1378]) {
      return local$$33063;
    } else {
      if (local$$33063[_0x34b6[1066]](0) != _0x34b6[1161] && local$$33214 != _0x34b6[1376]) {
        return local$$33056 + local$$33063;
      } else {
        if (local$$33063[_0x34b6[1066]](0) == _0x34b6[1161]) {
          return local$$33063;
        } else {
          if (local$$33214 == _0x34b6[1376]) {
            /** @type {number} */
            var local$$33274 = 0;
            do {
              local$$33274 = local$$33056[_0x34b6[382]](_0x34b6[1161]);
              local$$33056 = local$$33056[_0x34b6[702]](0, local$$33274);
              local$$33063 = local$$33063[_0x34b6[702]](3);
            } while (local$$33063[_0x34b6[702]](0, 3) == _0x34b6[1376]);
            return local$$33056 + _0x34b6[1161] + local$$33063;
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
LSJUtility[_0x34b6[1380]] = function() {
  /** @type {null} */
  var local$$33336 = null;
  if (window[_0x34b6[1381]]) {
    /** @type {!XMLHttpRequest} */
    local$$33336 = new XMLHttpRequest;
  } else {
    local$$33336 = new ActiveXObject(_0x34b6[1382]);
  }
  if (local$$33336 == null) {
    alert(_0x34b6[1383]);
  }
  return local$$33336;
};
/**
 * @return {?}
 */
LSJUtility[_0x34b6[1384]] = function() {
  /** @type {!Array} */
  var local$$33386 = new Array(_0x34b6[1385], _0x34b6[1386], _0x34b6[1387], _0x34b6[1388], _0x34b6[1389], _0x34b6[1390]);
  /** @type {number} */
  var local$$33389 = 0;
  for (; local$$33389 < local$$33386[_0x34b6[223]]; local$$33389++) {
    try {
      return new ActiveXObject(local$$33386[local$$33389]);
    } catch (local$$33401) {
      return document[_0x34b6[938]][_0x34b6[1391]](_0x34b6[381], _0x34b6[381], null);
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
  if (local$$33438 === _0x34b6[381]) {
    return undefined;
  }
  var local$$33464 = _0x34b6[1392] + local$$33440[_0x34b6[1393]]() + _0x34b6[1394] + local$$33440[_0x34b6[1395]]();
  /** @type {boolean} */
  var local$$33467 = true;
  /** @type {boolean} */
  var local$$33470 = true;
  var local$$33476 = local$$33440[_0x34b6[1396]]();
  var local$$33484 = document[_0x34b6[424]](_0x34b6[516]);
  /** @type {number} */
  local$$33484[_0x34b6[208]] = 1;
  /** @type {number} */
  local$$33484[_0x34b6[209]] = 1;
  local$$33484[_0x34b6[428]][_0x34b6[834]] = local$$33464;
  var local$$33512 = local$$33484[_0x34b6[403]](_0x34b6[402]);
  local$$33512[_0x34b6[834]] = local$$33464;
  local$$33512[_0x34b6[1397]] = _0x34b6[292];
  local$$33512[_0x34b6[572]] = local$$33476;
  local$$33512[_0x34b6[972]] = _0x34b6[656];
  local$$33484[_0x34b6[428]][_0x34b6[427]] = _0x34b6[429];
  document[_0x34b6[440]][_0x34b6[412]](local$$33484);
  var local$$33558 = measureText(local$$33512, local$$33438, local$$33467, local$$33470);
  local$$33558[_0x34b6[1398]] = Math[_0x34b6[532]](local$$33558[_0x34b6[208]], local$$33558[_0x34b6[669]][_0x34b6[1399]] - local$$33558[_0x34b6[669]][_0x34b6[1400]]);
  local$$33484[_0x34b6[1401]] = local$$33558;
  document[_0x34b6[440]][_0x34b6[529]](local$$33484);
  local$$33484[_0x34b6[428]][_0x34b6[427]] = _0x34b6[381];
  /** @type {number} */
  var local$$33609 = 0;
  if (local$$33439 != undefined) {
    local$$33609 = local$$33439[_0x34b6[208]];
    local$$33484[_0x34b6[209]] = local$$33439[_0x34b6[209]];
  } else {
    local$$33484[_0x34b6[209]] = local$$33558[_0x34b6[209]];
  }
  /** @type {number} */
  var local$$33644 = local$$33558[_0x34b6[209]] - local$$33558[_0x34b6[1402]];
  local$$33484[_0x34b6[208]] = local$$33558[_0x34b6[1398]] + local$$33609;
  /** @type {number} */
  var local$$33659 = local$$33484[_0x34b6[209]] - local$$33644;
  local$$33512[_0x34b6[834]] = local$$33464;
  local$$33512[_0x34b6[1397]] = _0x34b6[292];
  local$$33512[_0x34b6[572]] = local$$33476;
  if (local$$33467) {
    local$$33512[_0x34b6[983]] = local$$33440[_0x34b6[1404]]()[_0x34b6[1403]]();
    local$$33512[_0x34b6[1405]](local$$33438, local$$33609, local$$33659);
  }
  if (local$$33470) {
    local$$33512[_0x34b6[976]] = local$$33440[_0x34b6[1406]]()[_0x34b6[1403]]();
    local$$33512[_0x34b6[997]](local$$33438, local$$33609, local$$33659);
  }
  return local$$33484;
}
/**
 * @param {?} local$$33724
 * @param {?} local$$33725
 * @return {?}
 */
function getCSSValue(local$$33724, local$$33725) {
  return document[_0x34b6[397]][_0x34b6[703]](local$$33724, null)[_0x34b6[1407]](local$$33725);
}
/**
 * @param {?} local$$33742
 * @param {?} local$$33743
 * @param {boolean} local$$33744
 * @param {boolean} local$$33745
 * @return {?}
 */
function measureText(local$$33742, local$$33743, local$$33744, local$$33745) {
  var local$$33751 = local$$33742[_0x34b6[1408]](local$$33743);
  var local$$33759 = getCSSValue(local$$33742[_0x34b6[516]], _0x34b6[1409]);
  var local$$33775 = getCSSValue(local$$33742[_0x34b6[516]], _0x34b6[1410])[_0x34b6[626]](_0x34b6[450], _0x34b6[381]);
  /** @type {boolean} */
  var local$$33784 = !/\S/[_0x34b6[386]](local$$33743);
  local$$33751[_0x34b6[1411]] = local$$33775;
  var local$$33797 = document[_0x34b6[424]](_0x34b6[558]);
  local$$33797[_0x34b6[428]][_0x34b6[430]] = _0x34b6[451];
  /** @type {number} */
  local$$33797[_0x34b6[428]][_0x34b6[322]] = 0;
  local$$33797[_0x34b6[428]][_0x34b6[834]] = local$$33775 + _0x34b6[1394] + local$$33759;
  local$$33797[_0x34b6[785]] = local$$33743 + _0x34b6[1412] + local$$33743;
  document[_0x34b6[440]][_0x34b6[412]](local$$33797);
  /** @type {number} */
  local$$33751[_0x34b6[1413]] = 1.2 * local$$33775;
  var local$$33857 = getCSSValue(local$$33797, _0x34b6[209]);
  local$$33857 = local$$33857[_0x34b6[626]](_0x34b6[450], _0x34b6[381]);
  if (local$$33857 >= local$$33775 * 2) {
    /** @type {number} */
    local$$33751[_0x34b6[1413]] = local$$33857 / 2 | 0;
  }
  document[_0x34b6[440]][_0x34b6[529]](local$$33797);
  if (!local$$33784) {
    var local$$33899 = document[_0x34b6[424]](_0x34b6[516]);
    /** @type {number} */
    var local$$33902 = 100;
    local$$33899[_0x34b6[208]] = local$$33751[_0x34b6[208]] + local$$33902;
    /** @type {number} */
    local$$33899[_0x34b6[209]] = 3 * local$$33775;
    /** @type {number} */
    local$$33899[_0x34b6[428]][_0x34b6[322]] = 1;
    local$$33899[_0x34b6[428]][_0x34b6[562]] = local$$33759;
    local$$33899[_0x34b6[428]][_0x34b6[563]] = local$$33775;
    var local$$33951 = local$$33899[_0x34b6[403]](_0x34b6[402]);
    local$$33951[_0x34b6[834]] = local$$33775 + _0x34b6[1394] + local$$33759;
    var local$$33965 = local$$33899[_0x34b6[208]];
    var local$$33970 = local$$33899[_0x34b6[209]];
    /** @type {number} */
    var local$$33974 = local$$33970 / 2;
    local$$33951[_0x34b6[976]] = _0x34b6[1274];
    local$$33951[_0x34b6[977]](-1, -1, local$$33965 + 2, local$$33970 + 2);
    if (local$$33744) {
      local$$33951[_0x34b6[983]] = _0x34b6[1414];
      local$$33951[_0x34b6[572]] = local$$33742[_0x34b6[572]];
      local$$33951[_0x34b6[1405]](local$$33743, local$$33902 / 2, local$$33974);
    }
    if (local$$33745) {
      local$$33951[_0x34b6[976]] = _0x34b6[1414];
      local$$33951[_0x34b6[997]](local$$33743, local$$33902 / 2, local$$33974);
    }
    var local$$34045 = local$$33951[_0x34b6[401]](0, 0, local$$33965, local$$33970)[_0x34b6[575]];
    /** @type {number} */
    var local$$34048 = 0;
    /** @type {number} */
    var local$$34052 = local$$33965 * 4;
    var local$$34057 = local$$34045[_0x34b6[223]];
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
    local$$33751[_0x34b6[1402]] = local$$33974 - local$$34073;
    /** @type {number} */
    local$$33751[_0x34b6[1415]] = local$$34094 - local$$33974;
    local$$33751[_0x34b6[669]] = {
      minx : local$$34125 - local$$33902 / 2,
      maxx : local$$34167 - local$$33902 / 2,
      miny : 0,
      maxy : local$$34094 - local$$34073
    };
    /** @type {number} */
    local$$33751[_0x34b6[209]] = 1 + (local$$34094 - local$$34073);
  } else {
    /** @type {number} */
    local$$33751[_0x34b6[1402]] = 0;
    /** @type {number} */
    local$$33751[_0x34b6[1415]] = 0;
    local$$33751[_0x34b6[669]] = {
      minx : 0,
      maxx : local$$33751[_0x34b6[208]],
      miny : 0,
      maxy : 0
    };
    /** @type {number} */
    local$$33751[_0x34b6[209]] = 0;
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
  var local$$34257 = 1 / Math[_0x34b6[1416]](local$$34244 / 2) / local$$34245;
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
THREE[_0x34b6[1417]] = function(local$$34273, local$$34274, local$$34275) {
  /**
   * @param {?} local$$34278
   * @return {undefined}
   */
  function local$$34277(local$$34278) {
    local$$34278[_0x34b6[1428]]();
    /** @type {number} */
    local$$34285[_0x34b6[290]] = local$$34278[_0x34b6[1429]] / local$$34275[_0x34b6[208]] * 2 - 1;
    /** @type {number} */
    local$$34285[_0x34b6[291]] = -(local$$34278[_0x34b6[1430]] / local$$34275[_0x34b6[209]]) * 2 + 1;
    local$$34319[_0x34b6[1431]](local$$34285, local$$34273);
    var local$$34328 = local$$34319[_0x34b6[1360]];
    if (local$$34330 && local$$34331[_0x34b6[224]]) {
      var local$$34339 = local$$34330[_0x34b6[570]];
      var local$$34348 = local$$34339[_0x34b6[1432]](local$$34328[_0x34b6[353]]);
      if (local$$34348 == 0) {
        console[_0x34b6[514]](_0x34b6[1433]);
        return;
      }
      var local$$34382 = local$$34339[_0x34b6[1432]](local$$34366[_0x34b6[338]](local$$34330[_0x34b6[1435]])[_0x34b6[1434]](local$$34328[_0x34b6[602]]));
      /** @type {number} */
      var local$$34385 = local$$34382 / local$$34348;
      local$$34387[_0x34b6[338]](local$$34328[_0x34b6[353]])[_0x34b6[350]](local$$34385)[_0x34b6[274]](local$$34328[_0x34b6[602]])[_0x34b6[1434]](local$$34409);
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
        local$$34330[_0x34b6[368]][_0x34b6[430]][_0x34b6[290]] = local$$34387[_0x34b6[290]];
      }
      if (local$$34421) {
        local$$34330[_0x34b6[368]][_0x34b6[430]][_0x34b6[291]] = local$$34387[_0x34b6[291]];
      }
      if (local$$34423) {
        local$$34330[_0x34b6[368]][_0x34b6[430]][_0x34b6[1287]] = local$$34387[_0x34b6[1287]];
      }
      local$$34506(_0x34b6[1436], local$$34330);
      return;
    }
    local$$34319[_0x34b6[1431]](local$$34285, local$$34273);
    var local$$34524 = local$$34319[_0x34b6[1437]](local$$34274);
    if (local$$34524[_0x34b6[223]] > 0) {
      local$$34275[_0x34b6[428]][_0x34b6[1438]] = _0x34b6[1439];
      local$$34541 = local$$34524[0];
      local$$34506(_0x34b6[1440], local$$34541);
    } else {
      local$$34506(_0x34b6[1441], local$$34541);
      /** @type {null} */
      local$$34541 = null;
      local$$34275[_0x34b6[428]][_0x34b6[1438]] = _0x34b6[710];
    }
  }
  /**
   * @param {?} local$$34573
   * @return {undefined}
   */
  function local$$34572(local$$34573) {
    local$$34573[_0x34b6[1428]]();
    /** @type {number} */
    local$$34285[_0x34b6[290]] = local$$34573[_0x34b6[1429]] / local$$34275[_0x34b6[208]] * 2 - 1;
    /** @type {number} */
    local$$34285[_0x34b6[291]] = -(local$$34573[_0x34b6[1430]] / local$$34275[_0x34b6[209]]) * 2 + 1;
    local$$34319[_0x34b6[1431]](local$$34285, local$$34273);
    var local$$34622 = local$$34319[_0x34b6[1437]](local$$34274);
    var local$$34627 = local$$34319[_0x34b6[1360]];
    var local$$34632 = local$$34627[_0x34b6[353]];
    if (local$$34622[_0x34b6[223]] > 0) {
      local$$34330 = local$$34622[0];
      local$$34330[_0x34b6[1360]] = local$$34627;
      local$$34330[_0x34b6[570]] = local$$34632;
      local$$34409[_0x34b6[338]](local$$34330[_0x34b6[1435]])[_0x34b6[1434]](local$$34330[_0x34b6[368]][_0x34b6[430]]);
      local$$34275[_0x34b6[428]][_0x34b6[1438]] = _0x34b6[1442];
      local$$34506(_0x34b6[1443], local$$34330);
    }
  }
  /**
   * @param {?} local$$34690
   * @return {undefined}
   */
  function local$$34689(local$$34690) {
    local$$34690[_0x34b6[1428]]();
    if (local$$34330) {
      local$$34506(_0x34b6[1444], local$$34330);
      /** @type {null} */
      local$$34330 = null;
    }
    local$$34275[_0x34b6[428]][_0x34b6[1438]] = _0x34b6[710];
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
  this[_0x34b6[224]] = false;
  var local$$34752 = {};
  var local$$34331 = this;
  /**
   * @param {?} local$$34758
   * @param {?} local$$34759
   * @return {?}
   */
  this[_0x34b6[1170]] = function(local$$34758, local$$34759) {
    if (!local$$34752[local$$34758]) {
      /** @type {!Array} */
      local$$34752[local$$34758] = [];
    }
    local$$34752[local$$34758][_0x34b6[220]](local$$34759);
    return local$$34331;
  };
  /**
   * @param {?} local$$34784
   * @param {?} local$$34785
   * @return {?}
   */
  this[_0x34b6[1418]] = function(local$$34784, local$$34785) {
    var local$$34788 = local$$34752[local$$34784];
    if (!local$$34788) {
      return local$$34331;
    }
    if (local$$34788[_0x34b6[742]](local$$34785) > -1) {
      local$$34788[_0x34b6[222]](local$$34785, 1);
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
      for (; local$$34829 < local$$34820[_0x34b6[223]]; local$$34829++) {
        local$$34820[local$$34829](local$$34816);
      }
    }
  };
  /**
   * @param {?} local$$34850
   * @return {undefined}
   */
  this[_0x34b6[1419]] = function(local$$34850) {
    if (local$$34850 instanceof THREE[_0x34b6[1420]]) {
      local$$34274 = local$$34850[_0x34b6[684]];
    } else {
      local$$34274 = local$$34850;
    }
  };
  this[_0x34b6[1419]](local$$34274);
  /**
   * @return {undefined}
   */
  this[_0x34b6[1421]] = function() {
    local$$34275[_0x34b6[1423]](_0x34b6[1422], local$$34277, false);
    local$$34275[_0x34b6[1423]](_0x34b6[1424], local$$34572, false);
    local$$34275[_0x34b6[1423]](_0x34b6[1425], local$$34689, false);
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1426]] = function() {
    local$$34275[_0x34b6[1427]](_0x34b6[1422], local$$34277, false);
    local$$34275[_0x34b6[1427]](_0x34b6[1424], local$$34572, false);
    local$$34275[_0x34b6[1427]](_0x34b6[1425], local$$34689, false);
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[232]] = function() {
    local$$34331[_0x34b6[1426]]();
  };
  this[_0x34b6[1421]]();
};
/**
 * @return {undefined}
 */
LSJLayers = function() {
  /** @type {!Array} */
  this[_0x34b6[1445]] = [];
  this[_0x34b6[1446]] = new THREE.Group;
  this[_0x34b6[1447]] = new THREE.Sphere;
};
/** @type {function(): undefined} */
LSJLayers[_0x34b6[219]][_0x34b6[1183]] = LSJLayers;
/**
 * @return {undefined}
 */
LSJLayers[_0x34b6[219]][_0x34b6[232]] = function() {
  var local$$35009 = this[_0x34b6[1445]][_0x34b6[223]];
  /** @type {number} */
  var local$$35012 = 0;
  for (; local$$35012 < local$$35009; local$$35012++) {
    var local$$35021 = this[_0x34b6[1445]][local$$35012];
    this[_0x34b6[1446]][_0x34b6[1448]](local$$35021[_0x34b6[1446]]);
    if (local$$35021 != null) {
      local$$35021[_0x34b6[232]]();
    }
  }
  this[_0x34b6[1445]][_0x34b6[388]](0, local$$35009);
};
/**
 * @param {!Object} local$$35065
 * @return {undefined}
 */
LSJLayers[_0x34b6[219]][_0x34b6[1449]] = function(local$$35065) {
  if (local$$35065 == null || local$$35065 == undefined) {
    return;
  }
  this[_0x34b6[1445]][_0x34b6[220]](local$$35065);
  this[_0x34b6[1446]][_0x34b6[274]](local$$35065[_0x34b6[1446]]);
  LSJMath[_0x34b6[1451]](this[_0x34b6[1447]], local$$35065[_0x34b6[1450]]());
};
/**
 * @param {?} local$$35116
 * @return {?}
 */
LSJLayers[_0x34b6[219]][_0x34b6[1452]] = function(local$$35116) {
  var local$$35124 = this[_0x34b6[1445]][_0x34b6[223]];
  /** @type {number} */
  var local$$35127 = 0;
  for (; local$$35127 < local$$35124; local$$35127++) {
    var local$$35136 = this[_0x34b6[1445]][local$$35127];
    if (local$$35136[_0x34b6[1453]] == local$$35116) {
      return local$$35136;
    }
  }
  return null;
};
/**
 * @param {?} local$$35160
 * @return {?}
 */
LSJLayers[_0x34b6[219]][_0x34b6[1454]] = function(local$$35160) {
  var local$$35168 = this[_0x34b6[1445]][_0x34b6[223]];
  /** @type {number} */
  var local$$35171 = 0;
  for (; local$$35171 < local$$35168; local$$35171++) {
    var local$$35180 = local$$35160[_0x34b6[387]]();
    var local$$35186 = this[_0x34b6[1445]][local$$35171];
    if (local$$35186 != null) {
      var local$$35197 = local$$35186[_0x34b6[1115]][_0x34b6[387]]();
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
LSJLayers[_0x34b6[219]][_0x34b6[1455]] = function(local$$35220) {
  var local$$35228 = this[_0x34b6[1445]][_0x34b6[223]];
  if (local$$35220 >= 0 && local$$35220 < local$$35228) {
    return this[_0x34b6[1445]][local$$35220];
  }
  return null;
};
/**
 * @return {?}
 */
LSJLayers[_0x34b6[219]][_0x34b6[1450]] = function() {
  if (this[_0x34b6[1447]][_0x34b6[1456]]()) {
    var local$$35268 = this[_0x34b6[1445]][_0x34b6[223]];
    /** @type {number} */
    var local$$35271 = 0;
    for (; local$$35271 < local$$35268; local$$35271++) {
      var local$$35280 = this[_0x34b6[1445]][local$$35271];
      if (local$$35280 != null) {
        LSJMath[_0x34b6[1451]](this[_0x34b6[1447]], local$$35280[_0x34b6[1450]]());
      }
    }
  }
  return this[_0x34b6[1447]];
};
/**
 * @return {undefined}
 */
LSJLayers[_0x34b6[219]][_0x34b6[1457]] = function() {
  var local$$35324 = this[_0x34b6[1445]][_0x34b6[223]];
  /** @type {number} */
  var local$$35327 = 0;
  for (; local$$35327 < local$$35324; local$$35327++) {
    var local$$35336 = this[_0x34b6[1445]][local$$35327];
    if (local$$35336 != null && local$$35336[_0x34b6[445]] == _0x34b6[1458]) {
      local$$35336[_0x34b6[1457]]();
    }
  }
};
/**
 * @param {?} local$$35366
 * @return {undefined}
 */
LSJLayers[_0x34b6[219]][_0x34b6[225]] = function(local$$35366) {
  var local$$35374 = this[_0x34b6[1445]][_0x34b6[223]];
  /** @type {number} */
  var local$$35377 = 0;
  for (; local$$35377 < local$$35374; local$$35377++) {
    var local$$35386 = this[_0x34b6[1445]][local$$35377];
    if (local$$35386 != null) {
      local$$35386[_0x34b6[225]](local$$35366);
    }
  }
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1459]] = function() {
  /** @type {number} */
  this[_0x34b6[332]] = 0;
  /** @type {null} */
  this[_0x34b6[368]] = null;
  /** @type {number} */
  this[_0x34b6[1287]] = 0;
  /** @type {number} */
  this[_0x34b6[1460]] = 0;
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1461]] = function() {
  /** @type {number} */
  this[_0x34b6[332]] = 0;
  this[_0x34b6[1462]] = new THREE.RenderableVertex;
  this[_0x34b6[1463]] = new THREE.RenderableVertex;
  this[_0x34b6[1464]] = new THREE.RenderableVertex;
  this[_0x34b6[1465]] = new THREE.Vector3;
  /** @type {!Array} */
  this[_0x34b6[1466]] = [new THREE.Vector3, new THREE.Vector3, new THREE.Vector3];
  /** @type {number} */
  this[_0x34b6[1467]] = 0;
  this[_0x34b6[245]] = new THREE.Color;
  /** @type {null} */
  this[_0x34b6[268]] = null;
  /** @type {!Array} */
  this[_0x34b6[1130]] = [new THREE.Vector2, new THREE.Vector2, new THREE.Vector2];
  /** @type {number} */
  this[_0x34b6[1287]] = 0;
  /** @type {number} */
  this[_0x34b6[1460]] = 0;
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1468]] = function() {
  this[_0x34b6[430]] = new THREE.Vector3;
  this[_0x34b6[1469]] = new THREE.Vector3;
  this[_0x34b6[1470]] = new THREE.Vector4;
  /** @type {boolean} */
  this[_0x34b6[330]] = true;
};
/**
 * @param {?} local$$35579
 * @return {undefined}
 */
THREE[_0x34b6[1468]][_0x34b6[219]][_0x34b6[338]] = function(local$$35579) {
  this[_0x34b6[1469]][_0x34b6[338]](local$$35579[_0x34b6[1469]]);
  this[_0x34b6[1470]][_0x34b6[338]](local$$35579[_0x34b6[1470]]);
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1471]] = function() {
  /** @type {number} */
  this[_0x34b6[332]] = 0;
  this[_0x34b6[1462]] = new THREE.RenderableVertex;
  this[_0x34b6[1463]] = new THREE.RenderableVertex;
  /** @type {!Array} */
  this[_0x34b6[1472]] = [new THREE.Color, new THREE.Color];
  /** @type {null} */
  this[_0x34b6[268]] = null;
  /** @type {number} */
  this[_0x34b6[1287]] = 0;
  /** @type {number} */
  this[_0x34b6[1460]] = 0;
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1473]] = function() {
  /** @type {number} */
  this[_0x34b6[332]] = 0;
  /** @type {null} */
  this[_0x34b6[368]] = null;
  /** @type {number} */
  this[_0x34b6[290]] = 0;
  /** @type {number} */
  this[_0x34b6[291]] = 0;
  /** @type {number} */
  this[_0x34b6[1287]] = 0;
  /** @type {number} */
  this[_0x34b6[1271]] = 0;
  this[_0x34b6[1090]] = new THREE.Vector2;
  /** @type {null} */
  this[_0x34b6[268]] = null;
  /** @type {number} */
  this[_0x34b6[1460]] = 0;
};
/**
 * @return {undefined}
 */
THREE[_0x34b6[1474]] = function() {
  /**
   * @return {?}
   */
  function local$$35730() {
    if (local$$35732 === local$$35733) {
      var local$$35737 = new THREE.RenderableObject;
      local$$35739[_0x34b6[220]](local$$35737);
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
      local$$35767[_0x34b6[220]](local$$35765);
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
      local$$35795[_0x34b6[220]](local$$35793);
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
      local$$35823[_0x34b6[220]](local$$35821);
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
      local$$35851[_0x34b6[220]](local$$35849);
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
    if (local$$35871[_0x34b6[1460]] !== local$$35872[_0x34b6[1460]]) {
      return local$$35871[_0x34b6[1460]] - local$$35872[_0x34b6[1460]];
    } else {
      if (local$$35871[_0x34b6[1287]] !== local$$35872[_0x34b6[1287]]) {
        return local$$35872[_0x34b6[1287]] - local$$35871[_0x34b6[1287]];
      } else {
        if (local$$35871[_0x34b6[332]] !== local$$35872[_0x34b6[332]]) {
          return local$$35871[_0x34b6[332]] - local$$35872[_0x34b6[332]];
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
    var local$$35949 = local$$35933[_0x34b6[1287]] + local$$35933[_0x34b6[1484]];
    var local$$35958 = local$$35934[_0x34b6[1287]] + local$$35934[_0x34b6[1484]];
    var local$$35968 = -local$$35933[_0x34b6[1287]] + local$$35933[_0x34b6[1484]];
    var local$$35978 = -local$$35934[_0x34b6[1287]] + local$$35934[_0x34b6[1484]];
    if (local$$35949 >= 0 && local$$35958 >= 0 && local$$35968 >= 0 && local$$35978 >= 0) {
      return true;
    } else {
      if (local$$35949 < 0 && local$$35958 < 0 || local$$35968 < 0 && local$$35978 < 0) {
        return false;
      } else {
        if (local$$35949 < 0) {
          local$$35937 = Math[_0x34b6[532]](local$$35937, local$$35949 / (local$$35949 - local$$35958));
        } else {
          if (local$$35958 < 0) {
            local$$35940 = Math[_0x34b6[472]](local$$35940, local$$35949 / (local$$35949 - local$$35958));
          }
        }
        if (local$$35968 < 0) {
          local$$35937 = Math[_0x34b6[532]](local$$35937, local$$35968 / (local$$35968 - local$$35978));
        } else {
          if (local$$35978 < 0) {
            local$$35940 = Math[_0x34b6[472]](local$$35940, local$$35968 / (local$$35968 - local$$35978));
          }
        }
        if (local$$35940 < local$$35937) {
          return false;
        } else {
          local$$35933[_0x34b6[1526]](local$$35934, local$$35937);
          local$$35934[_0x34b6[1526]](local$$35933, 1 - local$$35940);
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
  this[_0x34b6[1475]] = function(local$$36197, local$$36198) {
    console[_0x34b6[1063]](_0x34b6[1476]);
    local$$36197[_0x34b6[1477]](local$$36198);
  };
  /**
   * @param {?} local$$36219
   * @param {?} local$$36220
   * @return {undefined}
   */
  this[_0x34b6[1478]] = function(local$$36219, local$$36220) {
    console[_0x34b6[1063]](_0x34b6[1479]);
    local$$36219[_0x34b6[1480]](local$$36220);
  };
  /**
   * @param {?} local$$36241
   * @param {?} local$$36242
   * @return {undefined}
   */
  this[_0x34b6[1481]] = function(local$$36241, local$$36242) {
    console[_0x34b6[217]](_0x34b6[1482]);
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
      local$$36266 = local$$36263[_0x34b6[268]];
      local$$36270[_0x34b6[1483]](local$$36263[_0x34b6[1285]]);
      /** @type {number} */
      local$$36257[_0x34b6[223]] = 0;
      /** @type {number} */
      local$$36260[_0x34b6[223]] = 0;
    };
    /**
     * @param {?} local$$36305
     * @return {undefined}
     */
    var local$$36404 = function(local$$36305) {
      var local$$36310 = local$$36305[_0x34b6[430]];
      var local$$36315 = local$$36305[_0x34b6[1469]];
      var local$$36320 = local$$36305[_0x34b6[1470]];
      local$$36315[_0x34b6[338]](local$$36310)[_0x34b6[1358]](local$$36170);
      local$$36320[_0x34b6[338]](local$$36315)[_0x34b6[1358]](local$$36168);
      /** @type {number} */
      var local$$36345 = 1 / local$$36320[_0x34b6[1484]];
      local$$36320[_0x34b6[290]] *= local$$36345;
      local$$36320[_0x34b6[291]] *= local$$36345;
      local$$36320[_0x34b6[1287]] *= local$$36345;
      /** @type {boolean} */
      local$$36305[_0x34b6[330]] = local$$36320[_0x34b6[290]] >= -1 && local$$36320[_0x34b6[290]] <= 1 && local$$36320[_0x34b6[291]] >= -1 && local$$36320[_0x34b6[291]] <= 1 && local$$36320[_0x34b6[1287]] >= -1 && local$$36320[_0x34b6[1287]] <= 1;
    };
    /**
     * @param {?} local$$36406
     * @param {?} local$$36407
     * @param {?} local$$36408
     * @return {undefined}
     */
    var local$$36425 = function(local$$36406, local$$36407, local$$36408) {
      local$$36093 = local$$35758();
      local$$36093[_0x34b6[430]][_0x34b6[334]](local$$36406, local$$36407, local$$36408);
      local$$36404(local$$36093);
    };
    /**
     * @param {?} local$$36427
     * @param {?} local$$36428
     * @param {?} local$$36429
     * @return {undefined}
     */
    var local$$36438 = function(local$$36427, local$$36428, local$$36429) {
      local$$36257[_0x34b6[220]](local$$36427, local$$36428, local$$36429);
    };
    /**
     * @param {?} local$$36440
     * @param {?} local$$36441
     * @return {undefined}
     */
    var local$$36450 = function(local$$36440, local$$36441) {
      local$$36260[_0x34b6[220]](local$$36440, local$$36441);
    };
    /**
     * @param {?} local$$36452
     * @param {?} local$$36453
     * @param {?} local$$36454
     * @return {?}
     */
    var local$$36510 = function(local$$36452, local$$36453, local$$36454) {
      if (local$$36452[_0x34b6[330]] === true || local$$36453[_0x34b6[330]] === true || local$$36454[_0x34b6[330]] === true) {
        return true;
      }
      local$$36156[0] = local$$36452[_0x34b6[1470]];
      local$$36156[1] = local$$36453[_0x34b6[1470]];
      local$$36156[2] = local$$36454[_0x34b6[1470]];
      return local$$36148[_0x34b6[1486]](local$$36152[_0x34b6[1485]](local$$36156));
    };
    /**
     * @param {?} local$$36512
     * @param {?} local$$36513
     * @param {?} local$$36514
     * @return {?}
     */
    var local$$36576 = function(local$$36512, local$$36513, local$$36514) {
      return (local$$36514[_0x34b6[1470]][_0x34b6[290]] - local$$36512[_0x34b6[1470]][_0x34b6[290]]) * (local$$36513[_0x34b6[1470]][_0x34b6[291]] - local$$36512[_0x34b6[1470]][_0x34b6[291]]) - (local$$36514[_0x34b6[1470]][_0x34b6[291]] - local$$36512[_0x34b6[1470]][_0x34b6[291]]) * (local$$36513[_0x34b6[1470]][_0x34b6[290]] - local$$36512[_0x34b6[1470]][_0x34b6[290]]) < 0;
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
      local$$36107[_0x34b6[332]] = local$$36263[_0x34b6[332]];
      local$$36107[_0x34b6[1462]][_0x34b6[338]](local$$36582);
      local$$36107[_0x34b6[1463]][_0x34b6[338]](local$$36585);
      /** @type {number} */
      local$$36107[_0x34b6[1287]] = (local$$36582[_0x34b6[1470]][_0x34b6[1287]] + local$$36585[_0x34b6[1470]][_0x34b6[1287]]) / 2;
      local$$36107[_0x34b6[1460]] = local$$36263[_0x34b6[1460]];
      local$$36107[_0x34b6[268]] = local$$36263[_0x34b6[268]];
      local$$36125[_0x34b6[1280]][_0x34b6[220]](local$$36107);
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
      if (local$$36266[_0x34b6[294]] === THREE[_0x34b6[295]] || local$$36576(local$$36667, local$$36670, local$$36673) === true) {
        local$$36100 = local$$35786();
        local$$36100[_0x34b6[332]] = local$$36263[_0x34b6[332]];
        local$$36100[_0x34b6[1462]][_0x34b6[338]](local$$36667);
        local$$36100[_0x34b6[1463]][_0x34b6[338]](local$$36670);
        local$$36100[_0x34b6[1464]][_0x34b6[338]](local$$36673);
        /** @type {number} */
        local$$36100[_0x34b6[1287]] = (local$$36667[_0x34b6[1470]][_0x34b6[1287]] + local$$36670[_0x34b6[1470]][_0x34b6[1287]] + local$$36673[_0x34b6[1470]][_0x34b6[1287]]) / 3;
        local$$36100[_0x34b6[1460]] = local$$36263[_0x34b6[1460]];
        local$$36100[_0x34b6[1465]][_0x34b6[462]](local$$36257, local$$36662 * 3);
        local$$36100[_0x34b6[1465]][_0x34b6[1488]](local$$36270)[_0x34b6[1487]]();
        /** @type {number} */
        var local$$36786 = 0;
        for (; local$$36786 < 3; local$$36786++) {
          var local$$36796 = local$$36100[_0x34b6[1466]][local$$36786];
          local$$36796[_0x34b6[462]](local$$36257, arguments[local$$36786] * 3);
          local$$36796[_0x34b6[1488]](local$$36270)[_0x34b6[1487]]();
          var local$$36819 = local$$36100[_0x34b6[1130]][local$$36786];
          local$$36819[_0x34b6[462]](local$$36260, arguments[local$$36786] * 2);
        }
        /** @type {number} */
        local$$36100[_0x34b6[1467]] = 3;
        local$$36100[_0x34b6[268]] = local$$36263[_0x34b6[268]];
        local$$36125[_0x34b6[1280]][_0x34b6[220]](local$$36100);
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
  this[_0x34b6[1489]] = function(local$$36873, local$$36874, local$$36875, local$$36876) {
    /** @type {number} */
    local$$35788 = 0;
    /** @type {number} */
    local$$35816 = 0;
    /** @type {number} */
    local$$35844 = 0;
    /** @type {number} */
    local$$36125[_0x34b6[1280]][_0x34b6[223]] = 0;
    if (local$$36873[_0x34b6[1490]] === true) {
      local$$36873[_0x34b6[1263]]();
    }
    if (local$$36874[_0x34b6[667]] === null) {
      local$$36874[_0x34b6[1263]]();
    }
    local$$36164[_0x34b6[338]](local$$36874[_0x34b6[337]][_0x34b6[1491]](local$$36874[_0x34b6[1285]]));
    local$$36168[_0x34b6[1286]](local$$36874[_0x34b6[335]], local$$36164);
    local$$36184[_0x34b6[1492]](local$$36168);
    /** @type {number} */
    local$$35732 = 0;
    /** @type {number} */
    local$$36125[_0x34b6[1128]][_0x34b6[223]] = 0;
    /** @type {number} */
    local$$36125[_0x34b6[1493]][_0x34b6[223]] = 0;
    local$$36873[_0x34b6[1500]](function(local$$36974) {
      if (local$$36974 instanceof THREE[_0x34b6[1494]]) {
        local$$36125[_0x34b6[1493]][_0x34b6[220]](local$$36974);
      } else {
        if (local$$36974 instanceof THREE[_0x34b6[329]] || local$$36974 instanceof THREE[_0x34b6[1137]] || local$$36974 instanceof THREE[_0x34b6[1495]]) {
          var local$$37006 = local$$36974[_0x34b6[268]];
          if (local$$37006[_0x34b6[330]] === false) {
            return;
          }
          if (local$$36974[_0x34b6[1496]] === false || local$$36184[_0x34b6[1497]](local$$36974) === true) {
            local$$36086 = local$$35730();
            local$$36086[_0x34b6[332]] = local$$36974[_0x34b6[332]];
            local$$36086[_0x34b6[368]] = local$$36974;
            local$$36129[_0x34b6[1498]](local$$36974[_0x34b6[1285]]);
            local$$36129[_0x34b6[1499]](local$$36168);
            local$$36086[_0x34b6[1287]] = local$$36129[_0x34b6[1287]];
            local$$36086[_0x34b6[1460]] = local$$36974[_0x34b6[1460]];
            local$$36125[_0x34b6[1128]][_0x34b6[220]](local$$36086);
          }
        }
      }
    });
    if (local$$36875 === true) {
      local$$36125[_0x34b6[1128]][_0x34b6[798]](local$$35870);
    }
    /** @type {number} */
    var local$$37106 = 0;
    var local$$37114 = local$$36125[_0x34b6[1128]][_0x34b6[223]];
    for (; local$$37106 < local$$37114; local$$37106++) {
      var local$$37126 = local$$36125[_0x34b6[1128]][local$$37106][_0x34b6[368]];
      var local$$37131 = local$$37126[_0x34b6[1126]];
      local$$36868[_0x34b6[1501]](local$$37126);
      local$$36170 = local$$37126[_0x34b6[1285]];
      /** @type {number} */
      local$$35760 = 0;
      if (local$$37126 instanceof THREE[_0x34b6[329]]) {
        if (local$$37131 instanceof THREE[_0x34b6[1502]]) {
          var local$$37157 = local$$37131[_0x34b6[1293]];
          var local$$37162 = local$$37131[_0x34b6[1503]];
          if (local$$37157[_0x34b6[430]] === undefined) {
            continue;
          }
          var local$$37178 = local$$37157[_0x34b6[430]][_0x34b6[1504]];
          /** @type {number} */
          var local$$37181 = 0;
          var local$$37186 = local$$37178[_0x34b6[223]];
          for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
            local$$36868[_0x34b6[1505]](local$$37178[local$$37181], local$$37178[local$$37181 + 1], local$$37178[local$$37181 + 2]);
          }
          if (local$$37157[_0x34b6[570]] !== undefined) {
            var local$$37218 = local$$37157[_0x34b6[570]][_0x34b6[1504]];
            /** @type {number} */
            local$$37181 = 0;
            local$$37186 = local$$37218[_0x34b6[223]];
            for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
              local$$36868[_0x34b6[1506]](local$$37218[local$$37181], local$$37218[local$$37181 + 1], local$$37218[local$$37181 + 2]);
            }
          }
          if (local$$37157[_0x34b6[1176]] !== undefined) {
            var local$$37260 = local$$37157[_0x34b6[1176]][_0x34b6[1504]];
            /** @type {number} */
            local$$37181 = 0;
            local$$37186 = local$$37260[_0x34b6[223]];
            for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 2) {
              local$$36868[_0x34b6[1507]](local$$37260[local$$37181], local$$37260[local$$37181 + 1]);
            }
          }
          if (local$$37131[_0x34b6[1120]] !== null) {
            var local$$37300 = local$$37131[_0x34b6[1120]][_0x34b6[1504]];
            if (local$$37162[_0x34b6[223]] > 0) {
              /** @type {number} */
              local$$37106 = 0;
              for (; local$$37106 < local$$37162[_0x34b6[223]]; local$$37106++) {
                var local$$37317 = local$$37162[local$$37106];
                local$$37181 = local$$37317[_0x34b6[495]];
                local$$37186 = local$$37317[_0x34b6[495]] + local$$37317[_0x34b6[1508]];
                for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
                  local$$36868[_0x34b6[1509]](local$$37300[local$$37181], local$$37300[local$$37181 + 1], local$$37300[local$$37181 + 2]);
                }
              }
            } else {
              /** @type {number} */
              local$$37181 = 0;
              local$$37186 = local$$37300[_0x34b6[223]];
              for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
                local$$36868[_0x34b6[1509]](local$$37300[local$$37181], local$$37300[local$$37181 + 1], local$$37300[local$$37181 + 2]);
              }
            }
          } else {
            /** @type {number} */
            local$$37181 = 0;
            /** @type {number} */
            local$$37186 = local$$37178[_0x34b6[223]] / 3;
            for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
              local$$36868[_0x34b6[1509]](local$$37181, local$$37181 + 1, local$$37181 + 2);
            }
          }
        } else {
          if (local$$37131 instanceof THREE[_0x34b6[1510]]) {
            var local$$37421 = local$$37131[_0x34b6[1125]];
            var local$$37426 = local$$37131[_0x34b6[1511]];
            var local$$37433 = local$$37131[_0x34b6[1512]][0];
            local$$36179[_0x34b6[1483]](local$$36170);
            var local$$37443 = local$$37126[_0x34b6[268]];
            /** @type {boolean} */
            var local$$37449 = local$$37443 instanceof THREE[_0x34b6[1513]];
            var local$$37458 = local$$37449 === true ? local$$37126[_0x34b6[268]] : null;
            /** @type {number} */
            var local$$37461 = 0;
            var local$$37466 = local$$37421[_0x34b6[223]];
            for (; local$$37461 < local$$37466; local$$37461++) {
              var local$$37472 = local$$37421[local$$37461];
              local$$36129[_0x34b6[338]](local$$37472);
              if (local$$37443[_0x34b6[1514]] === true) {
                var local$$37487 = local$$37131[_0x34b6[1514]];
                var local$$37492 = local$$37126[_0x34b6[1515]];
                /** @type {number} */
                var local$$37495 = 0;
                var local$$37500 = local$$37487[_0x34b6[223]];
                for (; local$$37495 < local$$37500; local$$37495++) {
                  var local$$37506 = local$$37492[local$$37495];
                  if (local$$37506 === 0) {
                    continue;
                  }
                  var local$$37515 = local$$37487[local$$37495];
                  var local$$37521 = local$$37515[_0x34b6[1125]][local$$37461];
                  local$$36129[_0x34b6[290]] += (local$$37521[_0x34b6[290]] - local$$37472[_0x34b6[290]]) * local$$37506;
                  local$$36129[_0x34b6[291]] += (local$$37521[_0x34b6[291]] - local$$37472[_0x34b6[291]]) * local$$37506;
                  local$$36129[_0x34b6[1287]] += (local$$37521[_0x34b6[1287]] - local$$37472[_0x34b6[1287]]) * local$$37506;
                }
              }
              local$$36868[_0x34b6[1505]](local$$36129[_0x34b6[290]], local$$36129[_0x34b6[291]], local$$36129[_0x34b6[1287]]);
            }
            /** @type {number} */
            var local$$37585 = 0;
            var local$$37590 = local$$37426[_0x34b6[223]];
            for (; local$$37585 < local$$37590; local$$37585++) {
              var local$$37596 = local$$37426[local$$37585];
              local$$37443 = local$$37449 === true ? local$$37458[_0x34b6[1078]][local$$37596[_0x34b6[1516]]] : local$$37126[_0x34b6[268]];
              if (local$$37443 === undefined) {
                continue;
              }
              var local$$37621 = local$$37443[_0x34b6[294]];
              var local$$37627 = local$$35767[local$$37596[_0x34b6[461]]];
              var local$$37633 = local$$35767[local$$37596[_0x34b6[460]]];
              var local$$37639 = local$$35767[local$$37596[_0x34b6[1020]]];
              if (local$$36868[_0x34b6[1517]](local$$37627, local$$37633, local$$37639) === false) {
                continue;
              }
              var local$$37655 = local$$36868[_0x34b6[1518]](local$$37627, local$$37633, local$$37639);
              if (local$$37621 !== THREE[_0x34b6[295]]) {
                if (local$$37621 === THREE[_0x34b6[1081]] && local$$37655 === false) {
                  continue;
                }
                if (local$$37621 === THREE[_0x34b6[1519]] && local$$37655 === true) {
                  continue;
                }
              }
              local$$36100 = local$$35786();
              local$$36100[_0x34b6[332]] = local$$37126[_0x34b6[332]];
              local$$36100[_0x34b6[1462]][_0x34b6[338]](local$$37627);
              local$$36100[_0x34b6[1463]][_0x34b6[338]](local$$37633);
              local$$36100[_0x34b6[1464]][_0x34b6[338]](local$$37639);
              local$$36100[_0x34b6[1465]][_0x34b6[338]](local$$37596[_0x34b6[570]]);
              if (local$$37655 === false && (local$$37621 === THREE[_0x34b6[1519]] || local$$37621 === THREE[_0x34b6[295]])) {
                local$$36100[_0x34b6[1465]][_0x34b6[1520]]();
              }
              local$$36100[_0x34b6[1465]][_0x34b6[1488]](local$$36179)[_0x34b6[1487]]();
              var local$$37769 = local$$37596[_0x34b6[1521]];
              /** @type {number} */
              var local$$37772 = 0;
              var local$$37782 = Math[_0x34b6[472]](local$$37769[_0x34b6[223]], 3);
              for (; local$$37772 < local$$37782; local$$37772++) {
                var local$$37791 = local$$36100[_0x34b6[1466]][local$$37772];
                local$$37791[_0x34b6[338]](local$$37769[local$$37772]);
                if (local$$37655 === false && (local$$37621 === THREE[_0x34b6[1519]] || local$$37621 === THREE[_0x34b6[295]])) {
                  local$$37791[_0x34b6[1520]]();
                }
                local$$37791[_0x34b6[1488]](local$$36179)[_0x34b6[1487]]();
              }
              local$$36100[_0x34b6[1467]] = local$$37769[_0x34b6[223]];
              var local$$37840 = local$$37433[local$$37585];
              if (local$$37840 !== undefined) {
                /** @type {number} */
                var local$$37844 = 0;
                for (; local$$37844 < 3; local$$37844++) {
                  local$$36100[_0x34b6[1130]][local$$37844][_0x34b6[338]](local$$37840[local$$37844]);
                }
              }
              local$$36100[_0x34b6[245]] = local$$37596[_0x34b6[245]];
              local$$36100[_0x34b6[268]] = local$$37443;
              /** @type {number} */
              local$$36100[_0x34b6[1287]] = (local$$37627[_0x34b6[1470]][_0x34b6[1287]] + local$$37633[_0x34b6[1470]][_0x34b6[1287]] + local$$37639[_0x34b6[1470]][_0x34b6[1287]]) / 3;
              local$$36100[_0x34b6[1460]] = local$$37126[_0x34b6[1460]];
              local$$36125[_0x34b6[1280]][_0x34b6[220]](local$$36100);
            }
          }
        }
      } else {
        if (local$$37126 instanceof THREE[_0x34b6[1137]]) {
          if (local$$37131 instanceof THREE[_0x34b6[1502]]) {
            local$$37157 = local$$37131[_0x34b6[1293]];
            if (local$$37157[_0x34b6[430]] !== undefined) {
              local$$37178 = local$$37157[_0x34b6[430]][_0x34b6[1504]];
              /** @type {number} */
              local$$37181 = 0;
              local$$37186 = local$$37178[_0x34b6[223]];
              for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 3) {
                local$$36868[_0x34b6[1505]](local$$37178[local$$37181], local$$37178[local$$37181 + 1], local$$37178[local$$37181 + 2]);
              }
              if (local$$37131[_0x34b6[1120]] !== null) {
                local$$37300 = local$$37131[_0x34b6[1120]][_0x34b6[1504]];
                /** @type {number} */
                local$$37181 = 0;
                local$$37186 = local$$37300[_0x34b6[223]];
                for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + 2) {
                  local$$36868[_0x34b6[1522]](local$$37300[local$$37181], local$$37300[local$$37181 + 1]);
                }
              } else {
                /** @type {number} */
                var local$$38026 = local$$37126 instanceof THREE[_0x34b6[1523]] ? 2 : 1;
                /** @type {number} */
                local$$37181 = 0;
                /** @type {number} */
                local$$37186 = local$$37178[_0x34b6[223]] / 3 - 1;
                for (; local$$37181 < local$$37186; local$$37181 = local$$37181 + local$$38026) {
                  local$$36868[_0x34b6[1522]](local$$37181, local$$37181 + 1);
                }
              }
            }
          } else {
            if (local$$37131 instanceof THREE[_0x34b6[1510]]) {
              local$$36174[_0x34b6[1286]](local$$36168, local$$36170);
              local$$37421 = local$$37126[_0x34b6[1126]][_0x34b6[1125]];
              if (local$$37421[_0x34b6[223]] === 0) {
                continue;
              }
              local$$37627 = local$$35758();
              local$$37627[_0x34b6[1470]][_0x34b6[338]](local$$37421[0])[_0x34b6[1358]](local$$36174);
              /** @type {number} */
              local$$38026 = local$$37126 instanceof THREE[_0x34b6[1523]] ? 2 : 1;
              /** @type {number} */
              local$$37461 = 1;
              local$$37466 = local$$37421[_0x34b6[223]];
              for (; local$$37461 < local$$37466; local$$37461++) {
                local$$37627 = local$$35758();
                local$$37627[_0x34b6[1470]][_0x34b6[338]](local$$37421[local$$37461])[_0x34b6[1358]](local$$36174);
                if ((local$$37461 + 1) % local$$38026 > 0) {
                  continue;
                }
                local$$37633 = local$$35767[local$$35760 - 2];
                local$$36188[_0x34b6[338]](local$$37627[_0x34b6[1470]]);
                local$$36192[_0x34b6[338]](local$$37633[_0x34b6[1470]]);
                if (local$$35932(local$$36188, local$$36192) === true) {
                  local$$36188[_0x34b6[350]](1 / local$$36188[_0x34b6[1484]]);
                  local$$36192[_0x34b6[350]](1 / local$$36192[_0x34b6[1484]]);
                  local$$36107 = local$$35814();
                  local$$36107[_0x34b6[332]] = local$$37126[_0x34b6[332]];
                  local$$36107[_0x34b6[1462]][_0x34b6[1470]][_0x34b6[338]](local$$36188);
                  local$$36107[_0x34b6[1463]][_0x34b6[1470]][_0x34b6[338]](local$$36192);
                  local$$36107[_0x34b6[1287]] = Math[_0x34b6[532]](local$$36188[_0x34b6[1287]], local$$36192[_0x34b6[1287]]);
                  local$$36107[_0x34b6[1460]] = local$$37126[_0x34b6[1460]];
                  local$$36107[_0x34b6[268]] = local$$37126[_0x34b6[268]];
                  if (local$$37126[_0x34b6[268]][_0x34b6[1472]] === THREE[_0x34b6[1524]]) {
                    local$$36107[_0x34b6[1472]][0][_0x34b6[338]](local$$37126[_0x34b6[1126]][_0x34b6[674]][local$$37461]);
                    local$$36107[_0x34b6[1472]][1][_0x34b6[338]](local$$37126[_0x34b6[1126]][_0x34b6[674]][local$$37461 - 1]);
                  }
                  local$$36125[_0x34b6[1280]][_0x34b6[220]](local$$36107);
                }
              }
            }
          }
        } else {
          if (local$$37126 instanceof THREE[_0x34b6[1495]]) {
            local$$36133[_0x34b6[334]](local$$36170[_0x34b6[1280]][12], local$$36170[_0x34b6[1280]][13], local$$36170[_0x34b6[1280]][14], 1);
            local$$36133[_0x34b6[1358]](local$$36168);
            /** @type {number} */
            var local$$38355 = 1 / local$$36133[_0x34b6[1484]];
            local$$36133[_0x34b6[1287]] *= local$$38355;
            if (local$$36133[_0x34b6[1287]] >= -1 && local$$36133[_0x34b6[1287]] <= 1) {
              local$$36114 = local$$35842();
              local$$36114[_0x34b6[332]] = local$$37126[_0x34b6[332]];
              /** @type {number} */
              local$$36114[_0x34b6[290]] = local$$36133[_0x34b6[290]] * local$$38355;
              /** @type {number} */
              local$$36114[_0x34b6[291]] = local$$36133[_0x34b6[291]] * local$$38355;
              local$$36114[_0x34b6[1287]] = local$$36133[_0x34b6[1287]];
              local$$36114[_0x34b6[1460]] = local$$37126[_0x34b6[1460]];
              local$$36114[_0x34b6[368]] = local$$37126;
              local$$36114[_0x34b6[1271]] = local$$37126[_0x34b6[1271]];
              /** @type {number} */
              local$$36114[_0x34b6[1090]][_0x34b6[290]] = local$$37126[_0x34b6[1090]][_0x34b6[290]] * Math[_0x34b6[1525]](local$$36114[_0x34b6[290]] - (local$$36133[_0x34b6[290]] + local$$36874[_0x34b6[335]][_0x34b6[1280]][0]) / (local$$36133[_0x34b6[1484]] + local$$36874[_0x34b6[335]][_0x34b6[1280]][12]));
              /** @type {number} */
              local$$36114[_0x34b6[1090]][_0x34b6[291]] = local$$37126[_0x34b6[1090]][_0x34b6[291]] * Math[_0x34b6[1525]](local$$36114[_0x34b6[291]] - (local$$36133[_0x34b6[291]] + local$$36874[_0x34b6[335]][_0x34b6[1280]][5]) / (local$$36133[_0x34b6[1484]] + local$$36874[_0x34b6[335]][_0x34b6[1280]][13]));
              local$$36114[_0x34b6[268]] = local$$37126[_0x34b6[268]];
              local$$36125[_0x34b6[1280]][_0x34b6[220]](local$$36114);
            }
          }
        }
      }
    }
    if (local$$36876 === true) {
      local$$36125[_0x34b6[1280]][_0x34b6[798]](local$$35870);
    }
    return local$$36125;
  };
};
(function() {
  _0x34b6[1527];
  /**
   * @param {?} local$$38582
   * @return {undefined}
   */
  var local$$38693 = function(local$$38582) {
    THREE[_0x34b6[1528]][_0x34b6[238]](this);
    /** @type {boolean} */
    this[_0x34b6[1307]] = false;
    /** @type {boolean} */
    this[_0x34b6[1309]] = false;
    this[_0x34b6[294]] = THREE[_0x34b6[1081]];
    /** @type {boolean} */
    this[_0x34b6[480]] = true;
    this[_0x34b6[1529]](local$$38582);
    this[_0x34b6[1530]] = this[_0x34b6[245]][_0x34b6[212]]();
    this[_0x34b6[1531]] = this[_0x34b6[322]];
    /**
     * @param {?} local$$38646
     * @return {undefined}
     */
    this[_0x34b6[1532]] = function(local$$38646) {
      if (local$$38646) {
        this[_0x34b6[245]][_0x34b6[1533]](1, 1, 0);
        /** @type {number} */
        this[_0x34b6[322]] = 1;
      } else {
        this[_0x34b6[245]][_0x34b6[338]](this[_0x34b6[1530]]);
        this[_0x34b6[322]] = this[_0x34b6[1531]];
      }
    };
  };
  local$$38693[_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1528]][_0x34b6[219]]);
  /** @type {function(?): undefined} */
  local$$38693[_0x34b6[219]][_0x34b6[1183]] = local$$38693;
  /**
   * @param {?} local$$38718
   * @return {undefined}
   */
  var local$$38827 = function(local$$38718) {
    THREE[_0x34b6[1177]][_0x34b6[238]](this);
    /** @type {boolean} */
    this[_0x34b6[1307]] = false;
    /** @type {boolean} */
    this[_0x34b6[1309]] = false;
    /** @type {boolean} */
    this[_0x34b6[480]] = true;
    /** @type {number} */
    this[_0x34b6[1534]] = 1;
    this[_0x34b6[1529]](local$$38718);
    this[_0x34b6[1530]] = this[_0x34b6[245]][_0x34b6[212]]();
    this[_0x34b6[1531]] = this[_0x34b6[322]];
    /**
     * @param {?} local$$38780
     * @return {undefined}
     */
    this[_0x34b6[1532]] = function(local$$38780) {
      if (local$$38780) {
        this[_0x34b6[245]][_0x34b6[1533]](1, 1, 0);
        /** @type {number} */
        this[_0x34b6[322]] = 1;
      } else {
        this[_0x34b6[245]][_0x34b6[338]](this[_0x34b6[1530]]);
        this[_0x34b6[322]] = this[_0x34b6[1531]];
      }
    };
  };
  local$$38827[_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1177]][_0x34b6[219]]);
  /** @type {function(?): undefined} */
  local$$38827[_0x34b6[219]][_0x34b6[1183]] = local$$38827;
  var local$$38856 = new local$$38693({
    visible : false,
    transparent : false
  });
  /**
   * @return {undefined}
   */
  THREE[_0x34b6[1535]] = function() {
    var local$$38862 = this;
    /**
     * @return {undefined}
     */
    this[_0x34b6[1253]] = function() {
      THREE[_0x34b6[1348]][_0x34b6[238]](this);
      this[_0x34b6[1536]] = new THREE.Object3D;
      this[_0x34b6[1537]] = new THREE.Object3D;
      this[_0x34b6[1538]] = new THREE.Object3D;
      this[_0x34b6[274]](this[_0x34b6[1536]]);
      this[_0x34b6[274]](this[_0x34b6[1537]]);
      this[_0x34b6[274]](this[_0x34b6[1538]]);
      var local$$38927 = new THREE.PlaneBufferGeometry(50, 50, 2, 2);
      var local$$38936 = new THREE.MeshBasicMaterial({
        visible : false,
        side : THREE[_0x34b6[295]]
      });
      var local$$38947 = {
        "XY" : new THREE.Mesh(local$$38927, local$$38936),
        "YZ" : new THREE.Mesh(local$$38927, local$$38936),
        "XZ" : new THREE.Mesh(local$$38927, local$$38936),
        "XYZE" : new THREE.Mesh(local$$38927, local$$38936)
      };
      this[_0x34b6[1539]] = local$$38947[_0x34b6[1540]];
      local$$38947[_0x34b6[1541]][_0x34b6[1271]][_0x34b6[334]](0, Math[_0x34b6[979]] / 2, 0);
      local$$38947[_0x34b6[1542]][_0x34b6[1271]][_0x34b6[334]](-Math[_0x34b6[979]] / 2, 0, 0);
      var local$$38994;
      for (local$$38994 in local$$38947) {
        local$$38947[local$$38994][_0x34b6[1115]] = local$$38994;
        this[_0x34b6[1538]][_0x34b6[274]](local$$38947[local$$38994]);
        this[_0x34b6[1538]][local$$38994] = local$$38947[local$$38994];
      }
      /**
       * @param {!Object} local$$39021
       * @param {?} local$$39022
       * @return {undefined}
       */
      var local$$39103 = function(local$$39021, local$$39022) {
        var local$$39024;
        for (local$$39024 in local$$39021) {
          local$$38994 = local$$39021[local$$39024][_0x34b6[223]];
          for (; local$$38994--;) {
            var local$$39039 = local$$39021[local$$39024][local$$38994][0];
            var local$$39045 = local$$39021[local$$39024][local$$38994][1];
            var local$$39051 = local$$39021[local$$39024][local$$38994][2];
            /** @type {string} */
            local$$39039[_0x34b6[1115]] = local$$39024;
            if (local$$39045) {
              local$$39039[_0x34b6[430]][_0x34b6[334]](local$$39045[0], local$$39045[1], local$$39045[2]);
            }
            if (local$$39051) {
              local$$39039[_0x34b6[1271]][_0x34b6[334]](local$$39051[0], local$$39051[1], local$$39051[2]);
            }
            local$$39022[_0x34b6[274]](local$$39039);
          }
        }
      };
      local$$39103(this[_0x34b6[1543]], this[_0x34b6[1536]]);
      local$$39103(this[_0x34b6[1544]], this[_0x34b6[1537]]);
      this[_0x34b6[331]](function(local$$39124) {
        if (local$$39124 instanceof THREE[_0x34b6[329]]) {
          local$$39124[_0x34b6[1545]]();
          var local$$39142 = local$$39124[_0x34b6[1126]][_0x34b6[212]]();
          local$$39142[_0x34b6[1546]](local$$39124[_0x34b6[740]]);
          local$$39124[_0x34b6[1126]] = local$$39142;
          local$$39124[_0x34b6[430]][_0x34b6[334]](0, 0, 0);
          local$$39124[_0x34b6[1271]][_0x34b6[334]](0, 0, 0);
          local$$39124[_0x34b6[1090]][_0x34b6[334]](1, 1, 1);
        }
      });
    };
    /**
     * @param {?} local$$39203
     * @return {undefined}
     */
    this[_0x34b6[1532]] = function(local$$39203) {
      this[_0x34b6[331]](function(local$$39208) {
        if (local$$39208[_0x34b6[268]] && local$$39208[_0x34b6[268]][_0x34b6[1532]]) {
          if (local$$39208[_0x34b6[1115]] === local$$39203) {
            local$$39208[_0x34b6[268]][_0x34b6[1532]](true);
          } else {
            local$$39208[_0x34b6[268]][_0x34b6[1532]](false);
          }
        }
      });
    };
  };
  THREE[_0x34b6[1535]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1348]][_0x34b6[219]]);
  THREE[_0x34b6[1535]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1535]];
  /**
   * @param {?} local$$39300
   * @param {?} local$$39301
   * @return {undefined}
   */
  THREE[_0x34b6[1535]][_0x34b6[219]][_0x34b6[1261]] = function(local$$39300, local$$39301) {
    var local$$39308 = new THREE.Vector3(0, 0, 0);
    var local$$39315 = new THREE.Vector3(0, 1, 0);
    var local$$39319 = new THREE.Matrix4;
    this[_0x34b6[331]](function(local$$39324) {
      if (local$$39324[_0x34b6[1115]][_0x34b6[1548]](_0x34b6[1547]) !== -1) {
        local$$39324[_0x34b6[1239]][_0x34b6[1550]](local$$39319[_0x34b6[1549]](local$$39301, local$$39308, local$$39315));
      } else {
        if (local$$39324[_0x34b6[1115]][_0x34b6[1548]](_0x34b6[1551]) !== -1 || local$$39324[_0x34b6[1115]][_0x34b6[1548]](_0x34b6[1552]) !== -1 || local$$39324[_0x34b6[1115]][_0x34b6[1548]](_0x34b6[1553]) !== -1) {
          local$$39324[_0x34b6[1239]][_0x34b6[1554]](local$$39300);
        }
      }
    });
  };
  /**
   * @return {undefined}
   */
  THREE[_0x34b6[1555]] = function() {
    THREE[_0x34b6[1535]][_0x34b6[238]](this);
    var local$$39419 = new THREE.Geometry;
    var local$$39432 = new THREE.Mesh(new THREE.CylinderGeometry(0, .05, .2, 12, 1, false));
    /** @type {number} */
    local$$39432[_0x34b6[430]][_0x34b6[291]] = .5;
    local$$39432[_0x34b6[1545]]();
    local$$39419[_0x34b6[1556]](local$$39432[_0x34b6[1126]], local$$39432[_0x34b6[740]]);
    var local$$39461 = new THREE.BufferGeometry;
    local$$39461[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute([0, 0, 0, 1, 0, 0], 3));
    var local$$39483 = new THREE.BufferGeometry;
    local$$39483[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute([0, 0, 0, 0, 1, 0], 3));
    var local$$39504 = new THREE.BufferGeometry;
    local$$39504[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute([0, 0, 0, 0, 0, 1], 3));
    this[_0x34b6[1543]] = {
      X : [[new THREE.Mesh(local$$39419, new local$$38693({
        color : 16711680
      })), [.5, 0, 0], [0, 0, -Math[_0x34b6[979]] / 2]], [new THREE.Line(local$$39461, new local$$38827({
        color : 16711680
      }))]],
      Y : [[new THREE.Mesh(local$$39419, new local$$38693({
        color : 65280
      })), [0, .5, 0]], [new THREE.Line(local$$39483, new local$$38827({
        color : 65280
      }))]],
      Z : [[new THREE.Mesh(local$$39419, new local$$38693({
        color : 255
      })), [0, 0, .5], [Math[_0x34b6[979]] / 2, 0, 0]], [new THREE.Line(local$$39504, new local$$38827({
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
      })), [0, .15, .15], [0, Math[_0x34b6[979]] / 2, 0]]],
      XZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.29, .29), new local$$38693({
        color : 16711935,
        opacity : .25
      })), [.15, 0, .15], [-Math[_0x34b6[979]] / 2, 0, 0]]]
    };
    this[_0x34b6[1544]] = {
      X : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [.6, 0, 0], [0, 0, -Math[_0x34b6[979]] / 2]]],
      Y : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, .6, 0]]],
      Z : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, 0, .6], [Math[_0x34b6[979]] / 2, 0, 0]]],
      XYZ : [[new THREE.Mesh(new THREE.OctahedronGeometry(.2, 0), local$$38856)]],
      XY : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.4, .4), local$$38856), [.2, .2, 0]]],
      YZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.4, .4), local$$38856), [0, .2, .2], [0, Math[_0x34b6[979]] / 2, 0]]],
      XZ : [[new THREE.Mesh(new THREE.PlaneBufferGeometry(.4, .4), local$$38856), [.2, 0, .2], [-Math[_0x34b6[979]] / 2, 0, 0]]]
    };
    /**
     * @param {?} local$$39818
     * @param {?} local$$39819
     * @return {undefined}
     */
    this[_0x34b6[1557]] = function(local$$39818, local$$39819) {
      var local$$39823 = new THREE.Matrix4;
      local$$39819[_0x34b6[1358]](local$$39823[_0x34b6[1491]](local$$39823[_0x34b6[1559]](this[_0x34b6[1538]][_0x34b6[1558]][_0x34b6[1285]])));
      if (local$$39818 === _0x34b6[1551]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1558]];
        if (Math[_0x34b6[1525]](local$$39819[_0x34b6[291]]) > Math[_0x34b6[1525]](local$$39819[_0x34b6[1287]])) {
          this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1542]];
        }
      }
      if (local$$39818 === _0x34b6[1552]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1558]];
        if (Math[_0x34b6[1525]](local$$39819[_0x34b6[290]]) > Math[_0x34b6[1525]](local$$39819[_0x34b6[1287]])) {
          this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1541]];
        }
      }
      if (local$$39818 === _0x34b6[1553]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1542]];
        if (Math[_0x34b6[1525]](local$$39819[_0x34b6[290]]) > Math[_0x34b6[1525]](local$$39819[_0x34b6[291]])) {
          this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1541]];
        }
      }
      if (local$$39818 === _0x34b6[1560]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1540]];
      }
      if (local$$39818 === _0x34b6[1558]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1558]];
      }
      if (local$$39818 === _0x34b6[1541]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1541]];
      }
      if (local$$39818 === _0x34b6[1542]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1542]];
      }
    };
    this[_0x34b6[1253]]();
  };
  THREE[_0x34b6[1555]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1535]][_0x34b6[219]]);
  THREE[_0x34b6[1555]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1555]];
  /**
   * @return {undefined}
   */
  THREE[_0x34b6[1561]] = function() {
    THREE[_0x34b6[1535]][_0x34b6[238]](this);
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
        if (local$$40107 === _0x34b6[290]) {
          local$$40115[_0x34b6[220]](0, Math[_0x34b6[349]](local$$40122 / 32 * Math[_0x34b6[979]]) * local$$40106, Math[_0x34b6[1562]](local$$40122 / 32 * Math[_0x34b6[979]]) * local$$40106);
        }
        if (local$$40107 === _0x34b6[291]) {
          local$$40115[_0x34b6[220]](Math[_0x34b6[349]](local$$40122 / 32 * Math[_0x34b6[979]]) * local$$40106, 0, Math[_0x34b6[1562]](local$$40122 / 32 * Math[_0x34b6[979]]) * local$$40106);
        }
        if (local$$40107 === _0x34b6[1287]) {
          local$$40115[_0x34b6[220]](Math[_0x34b6[1562]](local$$40122 / 32 * Math[_0x34b6[979]]) * local$$40106, Math[_0x34b6[349]](local$$40122 / 32 * Math[_0x34b6[979]]) * local$$40106, 0);
        }
      }
      local$$40112[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute(local$$40115, 3));
      return local$$40112;
    };
    this[_0x34b6[1543]] = {
      X : [[new THREE.Line(new local$$40246(1, _0x34b6[290], .5), new local$$38827({
        color : 16711680
      }))]],
      Y : [[new THREE.Line(new local$$40246(1, _0x34b6[291], .5), new local$$38827({
        color : 65280
      }))]],
      Z : [[new THREE.Line(new local$$40246(1, _0x34b6[1287], .5), new local$$38827({
        color : 255
      }))]],
      E : [[new THREE.Line(new local$$40246(1.25, _0x34b6[1287], 1), new local$$38827({
        color : 13421568
      }))]],
      XYZE : [[new THREE.Line(new local$$40246(1, _0x34b6[1287], 1), new local$$38827({
        color : 7895160
      }))]]
    };
    this[_0x34b6[1544]] = {
      X : [[new THREE.Mesh(new THREE.TorusGeometry(1, .12, 4, 12, Math.PI), local$$38856), [0, 0, 0], [0, -Math[_0x34b6[979]] / 2, -Math[_0x34b6[979]] / 2]]],
      Y : [[new THREE.Mesh(new THREE.TorusGeometry(1, .12, 4, 12, Math.PI), local$$38856), [0, 0, 0], [Math[_0x34b6[979]] / 2, 0, 0]]],
      Z : [[new THREE.Mesh(new THREE.TorusGeometry(1, .12, 4, 12, Math.PI), local$$38856), [0, 0, 0], [0, 0, -Math[_0x34b6[979]] / 2]]],
      E : [[new THREE.Mesh(new THREE.TorusGeometry(1.25, .12, 2, 24), local$$38856)]],
      XYZE : [[new THREE.Mesh(new THREE.Geometry)]]
    };
    /**
     * @param {?} local$$40416
     * @return {undefined}
     */
    this[_0x34b6[1557]] = function(local$$40416) {
      if (local$$40416 === _0x34b6[1547]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1540]];
      }
      if (local$$40416 === _0x34b6[1551]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1541]];
      }
      if (local$$40416 === _0x34b6[1552]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1542]];
      }
      if (local$$40416 === _0x34b6[1553]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1558]];
      }
    };
    /**
     * @param {?} local$$40492
     * @param {?} local$$40493
     * @return {undefined}
     */
    this[_0x34b6[1261]] = function(local$$40492, local$$40493) {
      THREE[_0x34b6[1535]][_0x34b6[219]][_0x34b6[1261]][_0x34b6[652]](this, arguments);
      var local$$40516 = {
        handles : this[_0x34b6[1536]],
        pickers : this[_0x34b6[1537]]
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
      var local$$40571 = local$$40493[_0x34b6[212]]();
      local$$40528[_0x34b6[338]](this[_0x34b6[1538]][_0x34b6[1558]][_0x34b6[1271]]);
      local$$40532[_0x34b6[1554]](local$$40528);
      local$$40520[_0x34b6[1563]](local$$40532)[_0x34b6[1491]](local$$40520);
      local$$40571[_0x34b6[1358]](local$$40520);
      this[_0x34b6[331]](function(local$$40609) {
        local$$40532[_0x34b6[1554]](local$$40528);
        if (local$$40609[_0x34b6[1115]] === _0x34b6[1551]) {
          local$$40557[_0x34b6[1565]](local$$40539, Math[_0x34b6[1564]](-local$$40571[_0x34b6[291]], local$$40571[_0x34b6[1287]]));
          local$$40532[_0x34b6[1566]](local$$40532, local$$40557);
          local$$40609[_0x34b6[1239]][_0x34b6[338]](local$$40532);
        }
        if (local$$40609[_0x34b6[1115]] === _0x34b6[1552]) {
          local$$40561[_0x34b6[1565]](local$$40546, Math[_0x34b6[1564]](local$$40571[_0x34b6[290]], local$$40571[_0x34b6[1287]]));
          local$$40532[_0x34b6[1566]](local$$40532, local$$40561);
          local$$40609[_0x34b6[1239]][_0x34b6[338]](local$$40532);
        }
        if (local$$40609[_0x34b6[1115]] === _0x34b6[1553]) {
          local$$40565[_0x34b6[1565]](local$$40553, Math[_0x34b6[1564]](local$$40571[_0x34b6[291]], local$$40571[_0x34b6[290]]));
          local$$40532[_0x34b6[1566]](local$$40532, local$$40565);
          local$$40609[_0x34b6[1239]][_0x34b6[338]](local$$40532);
        }
      });
    };
    this[_0x34b6[1253]]();
  };
  THREE[_0x34b6[1561]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1535]][_0x34b6[219]]);
  THREE[_0x34b6[1561]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1561]];
  /**
   * @return {undefined}
   */
  THREE[_0x34b6[1567]] = function() {
    THREE[_0x34b6[1535]][_0x34b6[238]](this);
    var local$$40790 = new THREE.Geometry;
    var local$$40800 = new THREE.Mesh(new THREE.BoxGeometry(.125, .125, .125));
    /** @type {number} */
    local$$40800[_0x34b6[430]][_0x34b6[291]] = .5;
    local$$40800[_0x34b6[1545]]();
    local$$40790[_0x34b6[1556]](local$$40800[_0x34b6[1126]], local$$40800[_0x34b6[740]]);
    var local$$40829 = new THREE.BufferGeometry;
    local$$40829[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute([0, 0, 0, 1, 0, 0], 3));
    var local$$40850 = new THREE.BufferGeometry;
    local$$40850[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute([0, 0, 0, 0, 1, 0], 3));
    var local$$40871 = new THREE.BufferGeometry;
    local$$40871[_0x34b6[1174]](_0x34b6[430], new THREE.Float32Attribute([0, 0, 0, 0, 0, 1], 3));
    this[_0x34b6[1543]] = {
      X : [[new THREE.Mesh(local$$40790, new local$$38693({
        color : 16711680
      })), [.5, 0, 0], [0, 0, -Math[_0x34b6[979]] / 2]], [new THREE.Line(local$$40829, new local$$38827({
        color : 16711680
      }))]],
      Y : [[new THREE.Mesh(local$$40790, new local$$38693({
        color : 65280
      })), [0, .5, 0]], [new THREE.Line(local$$40850, new local$$38827({
        color : 65280
      }))]],
      Z : [[new THREE.Mesh(local$$40790, new local$$38693({
        color : 255
      })), [0, 0, .5], [Math[_0x34b6[979]] / 2, 0, 0]], [new THREE.Line(local$$40871, new local$$38827({
        color : 255
      }))]],
      XYZ : [[new THREE.Mesh(new THREE.BoxGeometry(.125, .125, .125), new local$$38693({
        color : 16777215,
        opacity : .25
      }))]]
    };
    this[_0x34b6[1544]] = {
      X : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [.6, 0, 0], [0, 0, -Math[_0x34b6[979]] / 2]]],
      Y : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, .6, 0]]],
      Z : [[new THREE.Mesh(new THREE.CylinderGeometry(.2, 0, 1, 4, 1, false), local$$38856), [0, 0, .6], [Math[_0x34b6[979]] / 2, 0, 0]]],
      XYZ : [[new THREE.Mesh(new THREE.BoxGeometry(.4, .4, .4), local$$38856)]]
    };
    /**
     * @param {?} local$$41060
     * @param {?} local$$41061
     * @return {undefined}
     */
    this[_0x34b6[1557]] = function(local$$41060, local$$41061) {
      var local$$41065 = new THREE.Matrix4;
      local$$41061[_0x34b6[1358]](local$$41065[_0x34b6[1491]](local$$41065[_0x34b6[1559]](this[_0x34b6[1538]][_0x34b6[1558]][_0x34b6[1285]])));
      if (local$$41060 === _0x34b6[1551]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1558]];
        if (Math[_0x34b6[1525]](local$$41061[_0x34b6[291]]) > Math[_0x34b6[1525]](local$$41061[_0x34b6[1287]])) {
          this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1542]];
        }
      }
      if (local$$41060 === _0x34b6[1552]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1558]];
        if (Math[_0x34b6[1525]](local$$41061[_0x34b6[290]]) > Math[_0x34b6[1525]](local$$41061[_0x34b6[1287]])) {
          this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1541]];
        }
      }
      if (local$$41060 === _0x34b6[1553]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1542]];
        if (Math[_0x34b6[1525]](local$$41061[_0x34b6[290]]) > Math[_0x34b6[1525]](local$$41061[_0x34b6[291]])) {
          this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1541]];
        }
      }
      if (local$$41060 === _0x34b6[1560]) {
        this[_0x34b6[1539]] = this[_0x34b6[1538]][_0x34b6[1540]];
      }
    };
    this[_0x34b6[1253]]();
  };
  THREE[_0x34b6[1567]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1535]][_0x34b6[219]]);
  THREE[_0x34b6[1567]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1567]];
  /**
   * @param {?} local$$41288
   * @param {!Object} local$$41289
   * @return {undefined}
   */
  THREE[_0x34b6[1568]] = function(local$$41288, local$$41289) {
    /**
     * @param {?} local$$41292
     * @return {undefined}
     */
    function local$$41291(local$$41292) {
      if (local$$41294[_0x34b6[368]] === undefined || local$$41299 === true || local$$41292[_0x34b6[1594]] !== undefined && local$$41292[_0x34b6[1594]] !== 0) {
        return;
      }
      var local$$41327 = local$$41292[_0x34b6[1595]] ? local$$41292[_0x34b6[1595]][0] : local$$41292;
      var local$$41340 = local$$41329(local$$41327, local$$41330[local$$41331][_0x34b6[1537]][_0x34b6[684]]);
      /** @type {null} */
      var local$$41343 = null;
      if (local$$41340) {
        local$$41343 = local$$41340[_0x34b6[368]][_0x34b6[1115]];
        local$$41292[_0x34b6[1428]]();
      }
      if (local$$41294[_0x34b6[1574]] !== local$$41343) {
        local$$41294[_0x34b6[1574]] = local$$41343;
        local$$41294[_0x34b6[1261]]();
        local$$41294[_0x34b6[1589]](local$$41378);
      }
    }
    /**
     * @param {?} local$$41386
     * @return {undefined}
     */
    function local$$41385(local$$41386) {
      if (local$$41294[_0x34b6[368]] === undefined || local$$41299 === true || local$$41386[_0x34b6[1594]] !== undefined && local$$41386[_0x34b6[1594]] !== 0) {
        return;
      }
      var local$$41419 = local$$41386[_0x34b6[1595]] ? local$$41386[_0x34b6[1595]][0] : local$$41386;
      if (local$$41419[_0x34b6[1594]] === 0 || local$$41419[_0x34b6[1594]] === undefined) {
        var local$$41439 = local$$41329(local$$41419, local$$41330[local$$41331][_0x34b6[1537]][_0x34b6[684]]);
        if (local$$41439) {
          local$$41386[_0x34b6[1428]]();
          local$$41386[_0x34b6[1596]]();
          local$$41294[_0x34b6[1589]](local$$41454);
          local$$41294[_0x34b6[1574]] = local$$41439[_0x34b6[368]][_0x34b6[1115]];
          local$$41294[_0x34b6[1261]]();
          local$$41473[_0x34b6[338]](local$$41477)[_0x34b6[1434]](local$$41482)[_0x34b6[1487]]();
          local$$41330[local$$41331][_0x34b6[1557]](local$$41294[_0x34b6[1574]], local$$41473);
          var local$$41504 = local$$41329(local$$41419, [local$$41330[local$$41331][_0x34b6[1539]]]);
          if (local$$41504) {
            local$$41506[_0x34b6[338]](local$$41294[_0x34b6[368]][_0x34b6[430]]);
            local$$41518[_0x34b6[338]](local$$41294[_0x34b6[368]][_0x34b6[1090]]);
            local$$41530[_0x34b6[1559]](local$$41294[_0x34b6[368]][_0x34b6[740]]);
            local$$41542[_0x34b6[1559]](local$$41294[_0x34b6[368]][_0x34b6[1285]]);
            local$$41554[_0x34b6[1559]](local$$41294[_0x34b6[368]][_0x34b6[667]][_0x34b6[1285]]);
            local$$41569[_0x34b6[1597]](local$$41573[_0x34b6[1491]](local$$41294[_0x34b6[368]][_0x34b6[667]][_0x34b6[1285]]));
            local$$41589[_0x34b6[338]](local$$41504[_0x34b6[1435]]);
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
      if (local$$41294[_0x34b6[368]] === undefined || local$$41294[_0x34b6[1574]] === null || local$$41299 === false || local$$41611[_0x34b6[1594]] !== undefined && local$$41611[_0x34b6[1594]] !== 0) {
        return;
      }
      var local$$41650 = local$$41611[_0x34b6[1595]] ? local$$41611[_0x34b6[1595]][0] : local$$41611;
      var local$$41658 = local$$41329(local$$41650, [local$$41330[local$$41331][_0x34b6[1539]]]);
      if (local$$41658 === false) {
        return;
      }
      local$$41611[_0x34b6[1428]]();
      local$$41611[_0x34b6[1596]]();
      local$$41676[_0x34b6[338]](local$$41658[_0x34b6[1435]]);
      if (local$$41331 === _0x34b6[996]) {
        local$$41676[_0x34b6[1434]](local$$41589);
        local$$41676[_0x34b6[336]](local$$41569);
        if (local$$41294[_0x34b6[1571]] === _0x34b6[1588]) {
          local$$41676[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$41542));
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1551]) === -1) {
            /** @type {number} */
            local$$41676[_0x34b6[290]] = 0;
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1552]) === -1) {
            /** @type {number} */
            local$$41676[_0x34b6[291]] = 0;
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1553]) === -1) {
            /** @type {number} */
            local$$41676[_0x34b6[1287]] = 0;
          }
          local$$41676[_0x34b6[1358]](local$$41530);
          local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[338]](local$$41506);
          local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[274]](local$$41676);
        }
        if (local$$41294[_0x34b6[1571]] === _0x34b6[1572] || local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1560]) !== -1) {
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1551]) === -1) {
            /** @type {number} */
            local$$41676[_0x34b6[290]] = 0;
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1552]) === -1) {
            /** @type {number} */
            local$$41676[_0x34b6[291]] = 0;
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1553]) === -1) {
            /** @type {number} */
            local$$41676[_0x34b6[1287]] = 0;
          }
          local$$41676[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$41554));
          local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[338]](local$$41506);
          local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[274]](local$$41676);
        }
        if (local$$41294[_0x34b6[1569]] !== null) {
          if (local$$41294[_0x34b6[1571]] === _0x34b6[1588]) {
            local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$41542));
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1551]) !== -1) {
            /** @type {number} */
            local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[290]] = Math[_0x34b6[292]](local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[290]] / local$$41294[_0x34b6[1569]]) * local$$41294[_0x34b6[1569]];
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1552]) !== -1) {
            /** @type {number} */
            local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[291]] = Math[_0x34b6[292]](local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[291]] / local$$41294[_0x34b6[1569]]) * local$$41294[_0x34b6[1569]];
          }
          if (local$$41294[_0x34b6[1574]][_0x34b6[1548]](_0x34b6[1553]) !== -1) {
            /** @type {number} */
            local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[1287]] = Math[_0x34b6[292]](local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[1287]] / local$$41294[_0x34b6[1569]]) * local$$41294[_0x34b6[1569]];
          }
          if (local$$41294[_0x34b6[1571]] === _0x34b6[1588]) {
            local$$41294[_0x34b6[368]][_0x34b6[430]][_0x34b6[1358]](local$$41542);
          }
        }
      } else {
        if (local$$41331 === _0x34b6[1090]) {
          local$$41676[_0x34b6[1434]](local$$41589);
          local$$41676[_0x34b6[336]](local$$41569);
          if (local$$41294[_0x34b6[1571]] === _0x34b6[1588]) {
            if (local$$41294[_0x34b6[1574]] === _0x34b6[1560]) {
              /** @type {number} */
              local$$42129 = 1 + local$$41676[_0x34b6[291]] / 50;
              /** @type {number} */
              local$$41294[_0x34b6[368]][_0x34b6[1090]][_0x34b6[290]] = local$$41518[_0x34b6[290]] * local$$42129;
              /** @type {number} */
              local$$41294[_0x34b6[368]][_0x34b6[1090]][_0x34b6[291]] = local$$41518[_0x34b6[291]] * local$$42129;
              /** @type {number} */
              local$$41294[_0x34b6[368]][_0x34b6[1090]][_0x34b6[1287]] = local$$41518[_0x34b6[1287]] * local$$42129;
            } else {
              local$$41676[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$41542));
              if (local$$41294[_0x34b6[1574]] === _0x34b6[1551]) {
                /** @type {number} */
                local$$41294[_0x34b6[368]][_0x34b6[1090]][_0x34b6[290]] = local$$41518[_0x34b6[290]] * (1 + local$$41676[_0x34b6[290]] / 50);
              }
              if (local$$41294[_0x34b6[1574]] === _0x34b6[1552]) {
                /** @type {number} */
                local$$41294[_0x34b6[368]][_0x34b6[1090]][_0x34b6[291]] = local$$41518[_0x34b6[291]] * (1 + local$$41676[_0x34b6[291]] / 50);
              }
              if (local$$41294[_0x34b6[1574]] === _0x34b6[1553]) {
                /** @type {number} */
                local$$41294[_0x34b6[368]][_0x34b6[1090]][_0x34b6[1287]] = local$$41518[_0x34b6[1287]] * (1 + local$$41676[_0x34b6[1287]] / 50);
              }
            }
          }
        } else {
          if (local$$41331 === _0x34b6[1598]) {
            local$$41676[_0x34b6[1434]](local$$41482);
            local$$41676[_0x34b6[336]](local$$41569);
            local$$42304[_0x34b6[338]](local$$41589)[_0x34b6[1434]](local$$41482);
            local$$42304[_0x34b6[336]](local$$41569);
            if (local$$41294[_0x34b6[1574]] === _0x34b6[1547]) {
              local$$41676[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$42331));
              local$$42304[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$42331));
              local$$42344[_0x34b6[334]](Math[_0x34b6[1564]](local$$41676[_0x34b6[1287]], local$$41676[_0x34b6[291]]), Math[_0x34b6[1564]](local$$41676[_0x34b6[290]], local$$41676[_0x34b6[1287]]), Math[_0x34b6[1564]](local$$41676[_0x34b6[291]], local$$41676[_0x34b6[290]]));
              local$$42380[_0x34b6[334]](Math[_0x34b6[1564]](local$$42304[_0x34b6[1287]], local$$42304[_0x34b6[291]]), Math[_0x34b6[1564]](local$$42304[_0x34b6[290]], local$$42304[_0x34b6[1287]]), Math[_0x34b6[1564]](local$$42304[_0x34b6[291]], local$$42304[_0x34b6[290]]));
              local$$42416[_0x34b6[1550]](local$$41573[_0x34b6[1491]](local$$41554));
              local$$42426[_0x34b6[1565]](local$$41473, local$$42344[_0x34b6[1287]] - local$$42380[_0x34b6[1287]]);
              local$$42439[_0x34b6[1550]](local$$41542);
              local$$42416[_0x34b6[1566]](local$$42416, local$$42426);
              local$$42416[_0x34b6[1566]](local$$42416, local$$42439);
              local$$41294[_0x34b6[368]][_0x34b6[1239]][_0x34b6[338]](local$$42416);
            } else {
              if (local$$41294[_0x34b6[1574]] === _0x34b6[1540]) {
                local$$42426[_0x34b6[1554]](local$$41676[_0x34b6[212]]()[_0x34b6[1599]](local$$42304)[_0x34b6[1487]]());
                local$$42416[_0x34b6[1550]](local$$41573[_0x34b6[1491]](local$$41554));
                local$$42499[_0x34b6[1565]](local$$42426, -local$$41676[_0x34b6[212]]()[_0x34b6[1600]](local$$42304));
                local$$42439[_0x34b6[1550]](local$$41542);
                local$$42416[_0x34b6[1566]](local$$42416, local$$42499);
                local$$42416[_0x34b6[1566]](local$$42416, local$$42439);
                local$$41294[_0x34b6[368]][_0x34b6[1239]][_0x34b6[338]](local$$42416);
              } else {
                if (local$$41294[_0x34b6[1571]] === _0x34b6[1588]) {
                  local$$41676[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$41542));
                  local$$42304[_0x34b6[1358]](local$$41573[_0x34b6[1491]](local$$41542));
                  local$$42344[_0x34b6[334]](Math[_0x34b6[1564]](local$$41676[_0x34b6[1287]], local$$41676[_0x34b6[291]]), Math[_0x34b6[1564]](local$$41676[_0x34b6[290]], local$$41676[_0x34b6[1287]]), Math[_0x34b6[1564]](local$$41676[_0x34b6[291]], local$$41676[_0x34b6[290]]));
                  local$$42380[_0x34b6[334]](Math[_0x34b6[1564]](local$$42304[_0x34b6[1287]], local$$42304[_0x34b6[291]]), Math[_0x34b6[1564]](local$$42304[_0x34b6[290]], local$$42304[_0x34b6[1287]]), Math[_0x34b6[1564]](local$$42304[_0x34b6[291]], local$$42304[_0x34b6[290]]));
                  local$$42439[_0x34b6[1550]](local$$41530);
                  if (local$$41294[_0x34b6[1570]] !== null) {
                    local$$42499[_0x34b6[1565]](local$$42648, Math[_0x34b6[292]]((local$$42344[_0x34b6[290]] - local$$42380[_0x34b6[290]]) / local$$41294[_0x34b6[1570]]) * local$$41294[_0x34b6[1570]]);
                    local$$42670[_0x34b6[1565]](local$$42674, Math[_0x34b6[292]]((local$$42344[_0x34b6[291]] - local$$42380[_0x34b6[291]]) / local$$41294[_0x34b6[1570]]) * local$$41294[_0x34b6[1570]]);
                    local$$42696[_0x34b6[1565]](local$$42700, Math[_0x34b6[292]]((local$$42344[_0x34b6[1287]] - local$$42380[_0x34b6[1287]]) / local$$41294[_0x34b6[1570]]) * local$$41294[_0x34b6[1570]]);
                  } else {
                    local$$42499[_0x34b6[1565]](local$$42648, local$$42344[_0x34b6[290]] - local$$42380[_0x34b6[290]]);
                    local$$42670[_0x34b6[1565]](local$$42674, local$$42344[_0x34b6[291]] - local$$42380[_0x34b6[291]]);
                    local$$42696[_0x34b6[1565]](local$$42700, local$$42344[_0x34b6[1287]] - local$$42380[_0x34b6[1287]]);
                  }
                  if (local$$41294[_0x34b6[1574]] === _0x34b6[1551]) {
                    local$$42439[_0x34b6[1566]](local$$42439, local$$42499);
                  }
                  if (local$$41294[_0x34b6[1574]] === _0x34b6[1552]) {
                    local$$42439[_0x34b6[1566]](local$$42439, local$$42670);
                  }
                  if (local$$41294[_0x34b6[1574]] === _0x34b6[1553]) {
                    local$$42439[_0x34b6[1566]](local$$42439, local$$42696);
                  }
                  local$$41294[_0x34b6[368]][_0x34b6[1239]][_0x34b6[338]](local$$42439);
                } else {
                  if (local$$41294[_0x34b6[1571]] === _0x34b6[1572]) {
                    local$$42344[_0x34b6[334]](Math[_0x34b6[1564]](local$$41676[_0x34b6[1287]], local$$41676[_0x34b6[291]]), Math[_0x34b6[1564]](local$$41676[_0x34b6[290]], local$$41676[_0x34b6[1287]]), Math[_0x34b6[1564]](local$$41676[_0x34b6[291]], local$$41676[_0x34b6[290]]));
                    local$$42380[_0x34b6[334]](Math[_0x34b6[1564]](local$$42304[_0x34b6[1287]], local$$42304[_0x34b6[291]]), Math[_0x34b6[1564]](local$$42304[_0x34b6[290]], local$$42304[_0x34b6[1287]]), Math[_0x34b6[1564]](local$$42304[_0x34b6[291]], local$$42304[_0x34b6[290]]));
                    local$$42416[_0x34b6[1550]](local$$41573[_0x34b6[1491]](local$$41554));
                    if (local$$41294[_0x34b6[1570]] !== null) {
                      local$$42499[_0x34b6[1565]](local$$42648, Math[_0x34b6[292]]((local$$42344[_0x34b6[290]] - local$$42380[_0x34b6[290]]) / local$$41294[_0x34b6[1570]]) * local$$41294[_0x34b6[1570]]);
                      local$$42670[_0x34b6[1565]](local$$42674, Math[_0x34b6[292]]((local$$42344[_0x34b6[291]] - local$$42380[_0x34b6[291]]) / local$$41294[_0x34b6[1570]]) * local$$41294[_0x34b6[1570]]);
                      local$$42696[_0x34b6[1565]](local$$42700, Math[_0x34b6[292]]((local$$42344[_0x34b6[1287]] - local$$42380[_0x34b6[1287]]) / local$$41294[_0x34b6[1570]]) * local$$41294[_0x34b6[1570]]);
                    } else {
                      local$$42499[_0x34b6[1565]](local$$42648, local$$42344[_0x34b6[290]] - local$$42380[_0x34b6[290]]);
                      local$$42670[_0x34b6[1565]](local$$42674, local$$42344[_0x34b6[291]] - local$$42380[_0x34b6[291]]);
                      local$$42696[_0x34b6[1565]](local$$42700, local$$42344[_0x34b6[1287]] - local$$42380[_0x34b6[1287]]);
                    }
                    local$$42439[_0x34b6[1550]](local$$41542);
                    if (local$$41294[_0x34b6[1574]] === _0x34b6[1551]) {
                      local$$42416[_0x34b6[1566]](local$$42416, local$$42499);
                    }
                    if (local$$41294[_0x34b6[1574]] === _0x34b6[1552]) {
                      local$$42416[_0x34b6[1566]](local$$42416, local$$42670);
                    }
                    if (local$$41294[_0x34b6[1574]] === _0x34b6[1553]) {
                      local$$42416[_0x34b6[1566]](local$$42416, local$$42696);
                    }
                    local$$42416[_0x34b6[1566]](local$$42416, local$$42439);
                    local$$41294[_0x34b6[368]][_0x34b6[1239]][_0x34b6[338]](local$$42416);
                  }
                }
              }
            }
          }
        }
      }
      local$$41294[_0x34b6[1261]]();
      local$$41294[_0x34b6[1589]](local$$41378);
      local$$41294[_0x34b6[1589]](local$$43109);
    }
    /**
     * @param {?} local$$43115
     * @return {undefined}
     */
    function local$$43114(local$$43115) {
      if (local$$43115[_0x34b6[1594]] !== undefined && local$$43115[_0x34b6[1594]] !== 0) {
        return;
      }
      if (local$$41299 && local$$41294[_0x34b6[1574]] !== null) {
        local$$43137[_0x34b6[1601]] = local$$41331;
        local$$41294[_0x34b6[1589]](local$$43137);
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
      var local$$43165 = local$$41289[_0x34b6[809]]();
      /** @type {number} */
      var local$$43178 = (local$$43158[_0x34b6[1429]] - local$$43165[_0x34b6[432]]) / local$$43165[_0x34b6[208]];
      /** @type {number} */
      var local$$43191 = (local$$43158[_0x34b6[1430]] - local$$43165[_0x34b6[434]]) / local$$43165[_0x34b6[209]];
      local$$43193[_0x34b6[334]](local$$43178 * 2 - 1, -(local$$43191 * 2) + 1);
      local$$43208[_0x34b6[1431]](local$$43193, local$$41288);
      var local$$43219 = local$$43208[_0x34b6[1437]](local$$43159, true);
      return local$$43219[0] ? local$$43219[0] : false;
    }
    THREE[_0x34b6[1348]][_0x34b6[238]](this);
    local$$41289 = local$$41289 !== undefined ? local$$41289 : document;
    this[_0x34b6[368]] = undefined;
    /** @type {boolean} */
    this[_0x34b6[330]] = false;
    /** @type {null} */
    this[_0x34b6[1569]] = null;
    /** @type {null} */
    this[_0x34b6[1570]] = null;
    this[_0x34b6[1571]] = _0x34b6[1572];
    /** @type {number} */
    this[_0x34b6[1573]] = 1;
    /** @type {null} */
    this[_0x34b6[1574]] = null;
    var local$$41294 = this;
    var local$$41331 = _0x34b6[996];
    /** @type {boolean} */
    var local$$41299 = false;
    var local$$43292 = _0x34b6[1558];
    var local$$41330 = {
      "translate" : new THREE.TransformGizmoTranslate,
      "rotate" : new THREE.TransformGizmoRotate,
      "scale" : new THREE.TransformGizmoScale
    };
    var local$$43302;
    for (local$$43302 in local$$41330) {
      var local$$43305 = local$$41330[local$$43302];
      /** @type {boolean} */
      local$$43305[_0x34b6[330]] = local$$43302 === local$$41331;
      this[_0x34b6[274]](local$$43305);
    }
    var local$$41378 = {
      type : _0x34b6[1575]
    };
    var local$$41454 = {
      type : _0x34b6[1576]
    };
    var local$$43137 = {
      type : _0x34b6[1577],
      mode : local$$41331
    };
    var local$$43109 = {
      type : _0x34b6[1578]
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
    local$$41289[_0x34b6[1423]](_0x34b6[1424], local$$41385, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1579], local$$41385, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1422], local$$41291, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1580], local$$41291, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1422], local$$41610, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1580], local$$41610, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1425], local$$43114, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1581], local$$43114, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1582], local$$43114, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1583], local$$43114, false);
    local$$41289[_0x34b6[1423]](_0x34b6[1584], local$$43114, false);
    /**
     * @return {undefined}
     */
    this[_0x34b6[232]] = function() {
      local$$41289[_0x34b6[1427]](_0x34b6[1424], local$$41385);
      local$$41289[_0x34b6[1427]](_0x34b6[1579], local$$41385);
      local$$41289[_0x34b6[1427]](_0x34b6[1422], local$$41291);
      local$$41289[_0x34b6[1427]](_0x34b6[1580], local$$41291);
      local$$41289[_0x34b6[1427]](_0x34b6[1422], local$$41610);
      local$$41289[_0x34b6[1427]](_0x34b6[1580], local$$41610);
      local$$41289[_0x34b6[1427]](_0x34b6[1425], local$$43114);
      local$$41289[_0x34b6[1427]](_0x34b6[1581], local$$43114);
      local$$41289[_0x34b6[1427]](_0x34b6[1582], local$$43114);
      local$$41289[_0x34b6[1427]](_0x34b6[1583], local$$43114);
      local$$41289[_0x34b6[1427]](_0x34b6[1584], local$$43114);
    };
    /**
     * @param {?} local$$43613
     * @return {undefined}
     */
    this[_0x34b6[1585]] = function(local$$43613) {
      this[_0x34b6[368]] = local$$43613;
      /** @type {boolean} */
      this[_0x34b6[330]] = true;
      this[_0x34b6[1261]]();
    };
    /**
     * @return {undefined}
     */
    this[_0x34b6[1586]] = function() {
      this[_0x34b6[368]] = undefined;
      /** @type {boolean} */
      this[_0x34b6[330]] = false;
      /** @type {null} */
      this[_0x34b6[1574]] = null;
    };
    /**
     * @param {string} local$$43663
     * @return {undefined}
     */
    this[_0x34b6[1587]] = function(local$$43663) {
      local$$41331 = local$$43663 ? local$$43663 : local$$41331;
      if (local$$41331 === _0x34b6[1090]) {
        local$$41294[_0x34b6[1571]] = _0x34b6[1588];
      }
      var local$$43681;
      for (local$$43681 in local$$41330) {
        /** @type {boolean} */
        local$$41330[local$$43681][_0x34b6[330]] = local$$43681 === local$$41331;
      }
      this[_0x34b6[1261]]();
      local$$41294[_0x34b6[1589]](local$$41378);
    };
    /**
     * @param {?} local$$43710
     * @return {undefined}
     */
    this[_0x34b6[1590]] = function(local$$43710) {
      local$$41294[_0x34b6[1569]] = local$$43710;
    };
    /**
     * @param {?} local$$43724
     * @return {undefined}
     */
    this[_0x34b6[1591]] = function(local$$43724) {
      local$$41294[_0x34b6[1570]] = local$$43724;
    };
    /**
     * @param {?} local$$43738
     * @return {undefined}
     */
    this[_0x34b6[221]] = function(local$$43738) {
      local$$41294[_0x34b6[1573]] = local$$43738;
      this[_0x34b6[1261]]();
      local$$41294[_0x34b6[1589]](local$$41378);
    };
    /**
     * @param {?} local$$43762
     * @return {undefined}
     */
    this[_0x34b6[1592]] = function(local$$43762) {
      local$$41294[_0x34b6[1571]] = local$$43762;
      this[_0x34b6[1261]]();
      local$$41294[_0x34b6[1589]](local$$41378);
    };
    /**
     * @return {undefined}
     */
    this[_0x34b6[1261]] = function() {
      if (local$$41294[_0x34b6[368]] === undefined) {
        return;
      }
      local$$41294[_0x34b6[368]][_0x34b6[1263]]();
      local$$41482[_0x34b6[1498]](local$$41294[_0x34b6[368]][_0x34b6[1285]]);
      local$$43425[_0x34b6[1550]](local$$41573[_0x34b6[1559]](local$$41294[_0x34b6[368]][_0x34b6[1285]]));
      local$$41288[_0x34b6[1263]]();
      local$$41477[_0x34b6[1498]](local$$41288[_0x34b6[1285]]);
      local$$43435[_0x34b6[1550]](local$$41573[_0x34b6[1559]](local$$41288[_0x34b6[1285]]));
      /** @type {number} */
      local$$42129 = local$$41482[_0x34b6[1593]](local$$41477) / 6 * local$$41294[_0x34b6[1573]];
      this[_0x34b6[430]][_0x34b6[338]](local$$41482);
      this[_0x34b6[1090]][_0x34b6[334]](local$$42129, local$$42129, local$$42129);
      local$$41473[_0x34b6[338]](local$$41477)[_0x34b6[1434]](local$$41482)[_0x34b6[1487]]();
      if (local$$41294[_0x34b6[1571]] === _0x34b6[1588]) {
        local$$41330[local$$41331][_0x34b6[1261]](local$$43425, local$$41473);
      } else {
        if (local$$41294[_0x34b6[1571]] === _0x34b6[1572]) {
          local$$41330[local$$41331][_0x34b6[1261]](new THREE.Euler, local$$41473);
        }
      }
      local$$41330[local$$41331][_0x34b6[1532]](local$$41294[_0x34b6[1574]]);
    };
  };
  THREE[_0x34b6[1568]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1348]][_0x34b6[219]]);
  THREE[_0x34b6[1568]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[1568]];
})();
/**
 * @return {undefined}
 */
LSJSkyBox = function() {
  this[_0x34b6[239]] = undefined;
  this[_0x34b6[240]] = undefined;
  this[_0x34b6[1602]] = undefined;
  /** @type {!Array} */
  this[_0x34b6[1603]] = [_0x34b6[1604], _0x34b6[1605], _0x34b6[1606], _0x34b6[1607], _0x34b6[1608], _0x34b6[1609]];
};
/** @type {function(): undefined} */
LSJSkyBox[_0x34b6[219]][_0x34b6[1183]] = LSJSkyBox;
/**
 * @param {?} local$$44035
 * @param {?} local$$44036
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1610]] = function(local$$44035, local$$44036) {
  var local$$44048 = (new THREE.CubeTextureLoader)[_0x34b6[1060]](this[_0x34b6[1603]]);
  local$$44048[_0x34b6[1194]] = THREE[_0x34b6[1611]];
  getScene()[_0x34b6[343]] = local$$44048;
};
/**
 * @param {?} local$$44075
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1612]] = function(local$$44075) {
  this[_0x34b6[1603]][3] = local$$44075;
  this[_0x34b6[1610]]();
};
/**
 * @param {?} local$$44099
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1613]] = function(local$$44099) {
  this[_0x34b6[1603]][2] = local$$44099;
  this[_0x34b6[1610]]();
};
/**
 * @param {?} local$$44123
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1614]] = function(local$$44123) {
  this[_0x34b6[1603]][4] = local$$44123;
  this[_0x34b6[1610]]();
};
/**
 * @param {?} local$$44147
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1615]] = function(local$$44147) {
  this[_0x34b6[1603]][5] = local$$44147;
  this[_0x34b6[1610]]();
};
/**
 * @param {?} local$$44171
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1616]] = function(local$$44171) {
  this[_0x34b6[1603]][1] = local$$44171;
  this[_0x34b6[1610]]();
};
/**
 * @param {?} local$$44195
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1617]] = function(local$$44195) {
  this[_0x34b6[1603]][0] = local$$44195;
  this[_0x34b6[1610]]();
};
/**
 * @param {(boolean|number|string)} local$$44219
 * @param {(boolean|number|string)} local$$44220
 * @return {undefined}
 */
LSJSkyBox[_0x34b6[219]][_0x34b6[1618]] = function(local$$44219, local$$44220) {
  if (this[_0x34b6[240]]) {
    /** @type {number} */
    this[_0x34b6[240]][_0x34b6[1619]] = local$$44219 / local$$44220;
    this[_0x34b6[240]][_0x34b6[1620]]();
  }
};
var Detector = {
  canvas : !!window[_0x34b6[1621]],
  webgl : function() {
    try {
      var local$$44262 = document[_0x34b6[424]](_0x34b6[516]);
      return !!(window[_0x34b6[1622]] && (local$$44262[_0x34b6[403]](_0x34b6[1623]) || local$$44262[_0x34b6[403]](_0x34b6[1624])));
    } catch (local$$44285) {
      return false;
    }
  }(),
  workers : !!window[_0x34b6[1625]],
  fileapi : window[_0x34b6[1626]] && window[_0x34b6[1627]] && window[_0x34b6[1628]] && window[_0x34b6[1629]],
  getWebGLErrorMessage : function() {
    var local$$44324 = document[_0x34b6[424]](_0x34b6[558]);
    local$$44324[_0x34b6[332]] = _0x34b6[1630];
    local$$44324[_0x34b6[428]][_0x34b6[562]] = _0x34b6[1631];
    local$$44324[_0x34b6[428]][_0x34b6[563]] = _0x34b6[1632];
    local$$44324[_0x34b6[428]][_0x34b6[705]] = _0x34b6[570];
    local$$44324[_0x34b6[428]][_0x34b6[839]] = _0x34b6[658];
    local$$44324[_0x34b6[428]][_0x34b6[343]] = _0x34b6[1633];
    local$$44324[_0x34b6[428]][_0x34b6[245]] = _0x34b6[1634];
    local$$44324[_0x34b6[428]][_0x34b6[565]] = _0x34b6[1635];
    local$$44324[_0x34b6[428]][_0x34b6[208]] = _0x34b6[1636];
    local$$44324[_0x34b6[428]][_0x34b6[564]] = _0x34b6[1637];
    if (!this[_0x34b6[1623]]) {
      local$$44324[_0x34b6[785]] = window[_0x34b6[1622]] ? [_0x34b6[1638], _0x34b6[1639]][_0x34b6[2]](_0x34b6[1]) : [_0x34b6[1640], _0x34b6[1639]][_0x34b6[2]](_0x34b6[1]);
    }
    return local$$44324;
  },
  addGetWebGLMessage : function(local$$44464) {
    var local$$44466;
    var local$$44468;
    var local$$44470;
    local$$44464 = local$$44464 || {};
    local$$44466 = local$$44464[_0x34b6[667]] !== undefined ? local$$44464[_0x34b6[667]] : document[_0x34b6[440]];
    local$$44468 = local$$44464[_0x34b6[332]] !== undefined ? local$$44464[_0x34b6[332]] : _0x34b6[1641];
    local$$44470 = Detector[_0x34b6[1642]]();
    local$$44470[_0x34b6[332]] = local$$44468;
    local$$44466[_0x34b6[412]](local$$44470);
  }
};
if (typeof module === _0x34b6[368]) {
  module[_0x34b6[369]] = Detector;
}
/**
 * @return {undefined}
 */
LSJStyle = function() {
  this[_0x34b6[445]] = _0x34b6[624];
  /** @type {boolean} */
  this[_0x34b6[1643]] = true;
};
/** @type {function(): undefined} */
LSJStyle[_0x34b6[219]][_0x34b6[1183]] = LSJStyle;
/**
 * @return {?}
 */
LSJStyle[_0x34b6[219]][_0x34b6[1644]] = function() {
  return this[_0x34b6[445]];
};
/**
 * @return {undefined}
 */
LSJTextStyle = function() {
  this[_0x34b6[1645]] = _0x34b6[1646];
  /** @type {number} */
  this[_0x34b6[563]] = 32;
  /** @type {number} */
  this[_0x34b6[1090]] = 1;
  /** @type {boolean} */
  this[_0x34b6[706]] = true;
  /** @type {boolean} */
  this[_0x34b6[1647]] = false;
  /** @type {boolean} */
  this[_0x34b6[1648]] = true;
  this[_0x34b6[1649]] = new THREE.Color;
  this[_0x34b6[1650]] = new THREE.Color;
  this[_0x34b6[1650]][_0x34b6[1533]](0, 0, 0);
  /** @type {number} */
  this[_0x34b6[1651]] = 1;
};
LSJTextStyle[_0x34b6[219]] = Object[_0x34b6[242]](LSJStyle[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJTextStyle[_0x34b6[219]][_0x34b6[1183]] = LSJTextStyle;
/**
 * @return {?}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1395]] = function() {
  return this[_0x34b6[1645]];
};
/**
 * @param {?} local$$44694
 * @return {undefined}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1652]] = function(local$$44694) {
  this[_0x34b6[1645]] = local$$44694;
};
/**
 * @return {?}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1393]] = function() {
  return this[_0x34b6[563]];
};
/**
 * @param {?} local$$44726
 * @return {undefined}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1653]] = function(local$$44726) {
  this[_0x34b6[563]] = local$$44726;
};
/**
 * @return {?}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1406]] = function() {
  return this[_0x34b6[1649]];
};
/**
 * @param {?} local$$44758
 * @return {undefined}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1654]] = function(local$$44758) {
  this[_0x34b6[1649]][_0x34b6[338]](local$$44758);
};
/**
 * @return {?}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1404]] = function() {
  return this[_0x34b6[1650]];
};
/**
 * @param {?} local$$44793
 * @return {undefined}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1655]] = function(local$$44793) {
  this[_0x34b6[1650]][_0x34b6[338]](local$$44793);
};
/**
 * @return {?}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1396]] = function() {
  return this[_0x34b6[1651]];
};
/**
 * @param {?} local$$44828
 * @return {undefined}
 */
LSJTextStyle[_0x34b6[219]][_0x34b6[1656]] = function(local$$44828) {
  this[_0x34b6[1651]] = local$$44828;
};
/**
 * @return {undefined}
 */
LSJMarkerStyle = function() {
  LSJStyle[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1657];
  this[_0x34b6[1291]] = undefined;
  this[_0x34b6[1658]] = new THREE.Color;
  /** @type {number} */
  this[_0x34b6[1659]] = 10;
  /** @type {boolean} */
  this[_0x34b6[1660]] = true;
  /** @type {boolean} */
  this[_0x34b6[1661]] = true;
  /** @type {boolean} */
  this[_0x34b6[1662]] = true;
  this[_0x34b6[1663]] = new LSJTextStyle;
};
LSJMarkerStyle[_0x34b6[219]] = Object[_0x34b6[242]](LSJStyle[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1183]] = LSJMarkerStyle;
/**
 * @param {?} local$$44927
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1664]] = function(local$$44927) {
  this[_0x34b6[1658]] = local$$44927;
};
/**
 * @return {?}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1665]] = function() {
  return this[_0x34b6[1291]];
};
/**
 * @param {?} local$$44959
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1666]] = function(local$$44959) {
  this[_0x34b6[1291]] = local$$44959;
};
/**
 * @return {?}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1667]] = function() {
  return this[_0x34b6[1668]];
};
/**
 * @param {?} local$$44991
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1669]] = function(local$$44991) {
  this[_0x34b6[1668]] = local$$44991;
};
/**
 * @param {?} local$$45008
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1670]] = function(local$$45008) {
  this[_0x34b6[428]] = local$$45008;
};
/**
 * @param {?} local$$45025
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1671]] = function(local$$45025) {
  this[_0x34b6[1663]] = local$$45025;
};
/**
 * @return {?}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1672]] = function() {
  return this[_0x34b6[1663]];
};
/**
 * @param {?} local$$45057
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1673]] = function(local$$45057) {
  this[_0x34b6[1660]] = local$$45057;
};
/**
 * @return {?}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1674]] = function() {
  return this[_0x34b6[1660]];
};
/**
 * @param {?} local$$45089
 * @return {undefined}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1675]] = function(local$$45089) {
  this[_0x34b6[1661]] = local$$45089;
};
/**
 * @return {?}
 */
LSJMarkerStyle[_0x34b6[219]][_0x34b6[1676]] = function() {
  return this[_0x34b6[1661]];
};
/**
 * @return {undefined}
 */
LSJIconStyle = function() {
  LSJStyle[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1677];
  this[_0x34b6[1291]] = _0x34b6[381];
  this[_0x34b6[1658]] = new THREE.Color;
  /** @type {number} */
  this[_0x34b6[1668]] = 1;
};
LSJIconStyle[_0x34b6[219]] = Object[_0x34b6[242]](LSJStyle[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJIconStyle[_0x34b6[219]][_0x34b6[1183]] = LSJIconStyle;
/**
 * @return {?}
 */
LSJIconStyle[_0x34b6[219]][_0x34b6[1678]] = function() {
  return this[_0x34b6[1658]];
};
/**
 * @param {?} local$$45196
 * @return {undefined}
 */
LSJIconStyle[_0x34b6[219]][_0x34b6[1664]] = function(local$$45196) {
  this[_0x34b6[1658]] = local$$45196;
};
/**
 * @return {?}
 */
LSJIconStyle[_0x34b6[219]][_0x34b6[1665]] = function() {
  return this[_0x34b6[1291]];
};
/**
 * @param {?} local$$45228
 * @return {undefined}
 */
LSJIconStyle[_0x34b6[219]][_0x34b6[1666]] = function(local$$45228) {
  this[_0x34b6[1291]] = local$$45228;
};
/**
 * @return {?}
 */
LSJIconStyle[_0x34b6[219]][_0x34b6[1667]] = function() {
  return this[_0x34b6[1668]];
};
/**
 * @param {?} local$$45260
 * @return {undefined}
 */
LSJIconStyle[_0x34b6[219]][_0x34b6[1669]] = function(local$$45260) {
  this[_0x34b6[1668]] = local$$45260;
};
/**
 * @return {undefined}
 */
LSJGeometry = function() {
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {number} */
  this[_0x34b6[332]] = 0;
  this[_0x34b6[1115]] = _0x34b6[381];
  this[_0x34b6[1679]] = _0x34b6[381];
  this[_0x34b6[445]] = _0x34b6[624];
  /** @type {null} */
  this[_0x34b6[428]] = null;
  /** @type {boolean} */
  this[_0x34b6[330]] = true;
};
/** @type {function(): undefined} */
LSJGeometry[_0x34b6[219]][_0x34b6[1183]] = LSJGeometry;
/**
 * @return {undefined}
 */
LSJGeometry[_0x34b6[219]][_0x34b6[232]] = function() {
  for (; this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]] > 0;) {
    var local$$45361 = this[_0x34b6[1446]][_0x34b6[684]][0];
    this[_0x34b6[1446]][_0x34b6[1448]](local$$45361);
  }
};
/**
 * @return {?}
 */
LSJGeometry[_0x34b6[219]][_0x34b6[1680]] = function() {
  return this[_0x34b6[1115]];
};
/**
 * @return {?}
 */
LSJGeometry[_0x34b6[219]][_0x34b6[1644]] = function() {
  return this[_0x34b6[445]];
};
/**
 * @return {?}
 */
LSJGeometry[_0x34b6[219]][_0x34b6[1681]] = function() {
  return this[_0x34b6[332]];
};
/**
 * @return {?}
 */
LSJGeometry[_0x34b6[219]][_0x34b6[1403]] = function() {
  return this[_0x34b6[428]];
};
/**
 * @return {undefined}
 */
LSJGeoMarker = function() {
  LSJGeometry[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1682];
  this[_0x34b6[1683]] = _0x34b6[381];
  this[_0x34b6[430]] = new THREE.Vector3(0, 0, 0);
  this[_0x34b6[1684]] = undefined;
  /** @type {boolean} */
  this[_0x34b6[1685]] = true;
  this[_0x34b6[1686]] = new LSJRectangle(0, 0, 0, 0);
  /** @type {boolean} */
  this[_0x34b6[1687]] = true;
  /** @type {number} */
  this[_0x34b6[1688]] = 1;
};
LSJGeoMarker[_0x34b6[219]] = Object[_0x34b6[242]](LSJGeometry[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1183]] = LSJGeoMarker;
/**
 * @param {!Object} local$$45533
 * @return {undefined}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1670]] = function(local$$45533) {
  if (local$$45533 == null) {
    return;
  }
  /** @type {!Object} */
  this[_0x34b6[428]] = local$$45533;
  /** @type {boolean} */
  this[_0x34b6[1685]] = true;
};
/**
 * @param {?} local$$45562
 * @return {undefined}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1689]] = function(local$$45562) {
  this[_0x34b6[1115]] = local$$45562;
  /** @type {boolean} */
  this[_0x34b6[1685]] = true;
};
/**
 * @return {undefined}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1261]] = function() {
  this[_0x34b6[232]]();
  if (this[_0x34b6[428]][_0x34b6[1291]] != undefined) {
    this[_0x34b6[1683]] = this[_0x34b6[428]][_0x34b6[1291]];
    /** @type {!Image} */
    var local$$45610 = new Image;
    local$$45610[_0x34b6[551]] = this[_0x34b6[428]][_0x34b6[1291]];
    var local$$45623 = this;
    /**
     * @return {undefined}
     */
    local$$45610[_0x34b6[443]] = function() {
      var local$$45629 = undefined;
      if (local$$45623[_0x34b6[1115]] != _0x34b6[381] && local$$45623[_0x34b6[1687]]) {
        var local$$45648 = local$$45623[_0x34b6[428]][_0x34b6[1672]]();
        local$$45629 = writeTextAndImgToCanvas(local$$45623[_0x34b6[1115]], local$$45610, local$$45648);
        /** @type {number} */
        local$$45623[_0x34b6[1688]] = local$$45629[_0x34b6[208]] / local$$45629[_0x34b6[209]];
      } else {
        local$$45629 = document[_0x34b6[424]](_0x34b6[516]);
        /** @type {number} */
        local$$45629[_0x34b6[208]] = 128 * local$$45610[_0x34b6[209]] / local$$45610[_0x34b6[208]];
        /** @type {number} */
        local$$45629[_0x34b6[209]] = 128;
        if (local$$45623[_0x34b6[1115]] == _0x34b6[381]) {
          /** @type {number} */
          local$$45623[_0x34b6[1688]] = local$$45629[_0x34b6[208]] / local$$45629[_0x34b6[209]];
        }
      }
      var local$$45726 = local$$45629[_0x34b6[403]](_0x34b6[402]);
      local$$45726[_0x34b6[542]](local$$45610, 0, 0, local$$45629[_0x34b6[209]] * local$$45610[_0x34b6[208]] / local$$45610[_0x34b6[209]], local$$45629[_0x34b6[209]]);
      var local$$45756 = new THREE.SpriteMaterial({
        depthTest : false,
        map : new THREE.CanvasTexture(local$$45629)
      });
      local$$45623[_0x34b6[1684]] = new LSJBillboard(local$$45756, local$$45623);
      local$$45623[_0x34b6[1684]][_0x34b6[430]][_0x34b6[338]](local$$45623[_0x34b6[430]]);
      local$$45623[_0x34b6[1446]][_0x34b6[274]](local$$45623[_0x34b6[1684]]);
      /** @type {boolean} */
      local$$45623[_0x34b6[1685]] = false;
    };
  } else {
    var local$$45807 = this[_0x34b6[428]][_0x34b6[1672]]();
    var local$$45814 = writeTextAndImgToCanvas(this[_0x34b6[1115]], null, local$$45807);
    var local$$45821 = new THREE.SpriteMaterial({
      map : new THREE.CanvasTexture(local$$45814)
    });
    /** @type {number} */
    this[_0x34b6[1688]] = local$$45814[_0x34b6[208]] / local$$45814[_0x34b6[209]];
    this[_0x34b6[1684]] = new LSJBillboard(local$$45821, this);
    this[_0x34b6[1684]][_0x34b6[430]][_0x34b6[338]](this[_0x34b6[430]]);
    this[_0x34b6[1446]][_0x34b6[274]](this[_0x34b6[1684]]);
    /** @type {boolean} */
    this[_0x34b6[1685]] = false;
  }
};
/**
 * @param {?} local$$45884
 * @param {?} local$$45885
 * @param {?} local$$45886
 * @return {undefined}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1241]] = function(local$$45884, local$$45885, local$$45886) {
  this[_0x34b6[430]][_0x34b6[290]] = local$$45884;
  this[_0x34b6[430]][_0x34b6[291]] = local$$45885;
  this[_0x34b6[430]][_0x34b6[1287]] = local$$45886;
  if (this[_0x34b6[1684]] != undefined) {
    this[_0x34b6[1684]][_0x34b6[430]][_0x34b6[338]](this[_0x34b6[430]]);
  }
};
/**
 * @return {?}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1240]] = function() {
  return this[_0x34b6[1690]];
};
/**
 * @return {?}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1288]] = function() {
  return this[_0x34b6[1686]];
};
/**
 * @param {?} local$$45972
 * @return {undefined}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1292]] = function(local$$45972) {
  if (this[_0x34b6[1687]] != local$$45972) {
    /** @type {boolean} */
    this[_0x34b6[1685]] = true;
  }
  this[_0x34b6[1687]] = local$$45972;
};
/**
 * @return {?}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[1691]] = function() {
  return this[_0x34b6[1687]];
};
/**
 * @param {string} local$$46017
 * @return {undefined}
 */
LSJGeoMarker[_0x34b6[219]][_0x34b6[225]] = function(local$$46017) {
  if (this[_0x34b6[1685]]) {
    this[_0x34b6[1261]]();
  }
  if (this[_0x34b6[1684]] != undefined) {
    this[_0x34b6[1684]][_0x34b6[240]] = getCamera();
    this[_0x34b6[1684]][_0x34b6[1263]]();
    local$$46017[_0x34b6[1692]][_0x34b6[1263]]();
    local$$46017[_0x34b6[1692]][_0x34b6[1620]]();
    var local$$46073 = new THREE.Vector3(0, 0, 0);
    var local$$46080 = new THREE.Vector3(0, 0, 0);
    local$$46080[_0x34b6[338]](this[_0x34b6[1684]][_0x34b6[430]]);
    /** @type {number} */
    var local$$46094 = 1;
    /** @type {number} */
    var local$$46097 = 1;
    var local$$46107 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$46080);
    if (local$$46107 > local$$46017[_0x34b6[1447]][_0x34b6[1693]] && local$$46017[_0x34b6[1447]][_0x34b6[1693]] != 0) {
      var local$$46163 = getCamera()[_0x34b6[430]][_0x34b6[212]]()[_0x34b6[274]](getCamera()[_0x34b6[430]][_0x34b6[212]]()[_0x34b6[1434]](local$$46080)[_0x34b6[1487]]()[_0x34b6[350]](local$$46017[_0x34b6[1447]][_0x34b6[1693]]));
      var local$$46176 = local$$46163[_0x34b6[212]]()[_0x34b6[1477]](local$$46017[_0x34b6[1692]]);
      /** @type {number} */
      var local$$46195 = (local$$46176[_0x34b6[290]] + 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[545]];
      var local$$46202 = new THREE.Vector3(0, 1, 0);
      local$$46202[_0x34b6[1696]](local$$46017[_0x34b6[1692]][_0x34b6[1239]]);
      var local$$46230 = local$$46163[_0x34b6[212]]()[_0x34b6[274]](local$$46202)[_0x34b6[1477]](local$$46017[_0x34b6[1692]]);
      /** @type {number} */
      var local$$46250 = -(local$$46176[_0x34b6[291]] - 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      var local$$46270 = -(local$$46230[_0x34b6[291]] - 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      var local$$46279 = 1 / Math[_0x34b6[1525]](local$$46250 - local$$46270);
      local$$46094 = Math[_0x34b6[1525]](local$$46279);
      local$$46176 = local$$46080[_0x34b6[212]]()[_0x34b6[1477]](local$$46017[_0x34b6[1692]]);
      local$$46202 = new THREE.Vector3(0, 1, 0);
      local$$46202[_0x34b6[1696]](local$$46017[_0x34b6[1692]][_0x34b6[1239]]);
      local$$46230 = local$$46080[_0x34b6[212]]()[_0x34b6[274]](local$$46202)[_0x34b6[1477]](local$$46017[_0x34b6[1692]]);
      /** @type {number} */
      local$$46250 = -(local$$46176[_0x34b6[291]] - 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      local$$46270 = -(local$$46230[_0x34b6[291]] - 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      var local$$46382 = 1 / Math[_0x34b6[1525]](local$$46250 - local$$46270);
      if (local$$46382 > 2 * local$$46279) {
        /** @type {number} */
        local$$46094 = local$$46382 / 2;
      }
      /** @type {number} */
      local$$46097 = local$$46279 / local$$46382;
    } else {
      local$$46176 = local$$46080[_0x34b6[212]]()[_0x34b6[1477]](local$$46017[_0x34b6[1692]]);
      local$$46202 = new THREE.Vector3(0, 1, 0);
      local$$46202[_0x34b6[1696]](local$$46017[_0x34b6[1692]][_0x34b6[1239]]);
      local$$46230 = local$$46080[_0x34b6[212]]()[_0x34b6[274]](local$$46202)[_0x34b6[1477]](local$$46017[_0x34b6[1692]]);
      /** @type {number} */
      local$$46250 = -(local$$46176[_0x34b6[291]] - 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      local$$46270 = -(local$$46230[_0x34b6[291]] - 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      local$$46094 = 1 / Math[_0x34b6[1525]](local$$46250 - local$$46270);
      /** @type {number} */
      local$$46195 = (local$$46176[_0x34b6[290]] + 1) / 2 * local$$46017[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[545]];
    }
    if (this[_0x34b6[428]][_0x34b6[1291]] != undefined) {
      /** @type {number} */
      this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[290]] = this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[291]] = this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[1287]] = this[_0x34b6[428]][_0x34b6[1659]] * local$$46094;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[432]] = local$$46195;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[656]] = local$$46250;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[434]] = local$$46250 + this[_0x34b6[428]][_0x34b6[1659]] * local$$46097;
    } else {
      var local$$46602 = this[_0x34b6[428]][_0x34b6[1672]]();
      /** @type {number} */
      this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[290]] = this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[291]] = this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[1287]] = local$$46602[_0x34b6[1393]]() * local$$46094;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[432]] = local$$46195;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[656]] = local$$46250;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[434]] = local$$46250 + local$$46602[_0x34b6[1393]]() * local$$46097;
    }
    /** @type {number} */
    var local$$46686 = this[_0x34b6[1686]][_0x34b6[434]] - this[_0x34b6[1686]][_0x34b6[656]];
    /** @type {number} */
    var local$$46692 = local$$46686 * this[_0x34b6[1688]];
    this[_0x34b6[1686]][_0x34b6[655]] = this[_0x34b6[1686]][_0x34b6[432]] + local$$46692;
    if (local$$46017 != undefined) {
      local$$46017[_0x34b6[1276]][_0x34b6[220]](this[_0x34b6[1684]]);
    }
  }
};
/**
 * @return {undefined}
 */
LSJMath = function() {
};
/** @type {function(): undefined} */
LSJMath[_0x34b6[219]][_0x34b6[1183]] = LSJMath;
/**
 * @param {?} local$$46747
 * @param {?} local$$46748
 * @return {undefined}
 */
LSJMath[_0x34b6[1451]] = function(local$$46747, local$$46748) {
  if (local$$46748[_0x34b6[1456]]()) {
    return;
  }
  if (local$$46747[_0x34b6[1456]]()) {
    local$$46747[_0x34b6[334]](local$$46748[_0x34b6[658]], local$$46748[_0x34b6[1693]]);
    return;
  }
  var local$$46779 = new THREE.Vector3;
  local$$46779[_0x34b6[1697]](local$$46747[_0x34b6[658]], local$$46748[_0x34b6[658]]);
  var local$$46796 = local$$46779[_0x34b6[223]]();
  if (local$$46796 + local$$46748[_0x34b6[1693]] <= local$$46747[_0x34b6[1693]]) {
    return;
  }
  if (local$$46796 + local$$46747[_0x34b6[1693]] <= local$$46748[_0x34b6[1693]]) {
    local$$46747[_0x34b6[334]](local$$46748[_0x34b6[658]], local$$46748[_0x34b6[1693]]);
    return;
  }
  /** @type {number} */
  var local$$46843 = (local$$46747[_0x34b6[1693]] + local$$46796 + local$$46748[_0x34b6[1693]]) * .5;
  /** @type {number} */
  var local$$46850 = (local$$46843 - local$$46747[_0x34b6[1693]]) / local$$46796;
  var local$$46872 = new THREE.Vector3(local$$46747[_0x34b6[658]][_0x34b6[290]], local$$46747[_0x34b6[658]][_0x34b6[291]], local$$46747[_0x34b6[658]][_0x34b6[1287]]);
  local$$46872[_0x34b6[290]] += (local$$46748[_0x34b6[658]][_0x34b6[290]] - local$$46747[_0x34b6[658]][_0x34b6[290]]) * local$$46850;
  local$$46872[_0x34b6[291]] += (local$$46748[_0x34b6[658]][_0x34b6[291]] - local$$46747[_0x34b6[658]][_0x34b6[291]]) * local$$46850;
  local$$46872[_0x34b6[290]] += (local$$46748[_0x34b6[658]][_0x34b6[1287]] - local$$46747[_0x34b6[658]][_0x34b6[1287]]) * local$$46850;
  local$$46747[_0x34b6[334]](local$$46872, local$$46843);
};
/**
 * @param {?} local$$46943
 * @param {?} local$$46944
 * @return {?}
 */
LSJMath[_0x34b6[1698]] = function(local$$46943, local$$46944) {
  /** @type {number} */
  var local$$46953 = local$$46943[_0x34b6[290]] - local$$46944[_0x34b6[290]];
  /** @type {number} */
  var local$$46962 = local$$46943[_0x34b6[291]] - local$$46944[_0x34b6[291]];
  /** @type {number} */
  var local$$46971 = local$$46943[_0x34b6[1287]] - local$$46944[_0x34b6[1287]];
  return Math[_0x34b6[889]](local$$46953 * local$$46953 + local$$46962 * local$$46962 + local$$46971 * local$$46971);
};
/**
 * @param {?} local$$46990
 * @param {?} local$$46991
 * @return {?}
 */
LSJMath[_0x34b6[1699]] = function(local$$46990, local$$46991) {
  /** @type {number} */
  var local$$47000 = local$$46990[_0x34b6[290]] - local$$46991[_0x34b6[290]];
  /** @type {number} */
  var local$$47009 = local$$46990[_0x34b6[291]] - local$$46991[_0x34b6[291]];
  /** @type {number} */
  var local$$47018 = local$$46990[_0x34b6[1287]] - local$$46991[_0x34b6[1287]];
  return local$$47000 * local$$47000 + local$$47009 * local$$47009 + local$$47018 * local$$47018;
};
/**
 * @param {?} local$$47033
 * @param {?} local$$47034
 * @return {?}
 */
LSJMath[_0x34b6[1700]] = function(local$$47033, local$$47034) {
  var local$$47040 = local$$47033[_0x34b6[1358]](local$$47034);
  return local$$47040[_0x34b6[223]]();
};
/**
 * @param {?} local$$47054
 * @return {?}
 */
LSJMath[_0x34b6[1361]] = function(local$$47054) {
  if (!LSJMath[_0x34b6[1362]](local$$47054[_0x34b6[208]]) || !LSJMath[_0x34b6[1362]](local$$47054[_0x34b6[209]])) {
    var local$$47079 = document[_0x34b6[424]](_0x34b6[516]);
    local$$47079[_0x34b6[208]] = LSJMath.Fs(local$$47054[_0x34b6[208]]);
    local$$47079[_0x34b6[209]] = LSJMath.Fs(local$$47054[_0x34b6[209]]);
    local$$47079[_0x34b6[403]](_0x34b6[402])[_0x34b6[542]](local$$47054, 0, 0, local$$47054[_0x34b6[208]], local$$47054[_0x34b6[209]], 0, 0, local$$47079[_0x34b6[208]], local$$47079[_0x34b6[209]]);
    return local$$47079;
  }
  return local$$47054;
};
/**
 * @param {number} local$$47140
 * @return {?}
 */
LSJMath[_0x34b6[1362]] = function(local$$47140) {
  return 0 === (local$$47140 & local$$47140 - 1);
};
/**
 * @param {number} local$$47155
 * @return {?}
 */
LSJMath[_0x34b6[1701]] = function(local$$47155) {
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
LSJMath[_0x34b6[1702]] = function(local$$47185, local$$47186) {
  return local$$47185[_0x34b6[290]] * local$$47186[_0x34b6[290]] + local$$47185[_0x34b6[291]] * local$$47186[_0x34b6[291]] + local$$47185[_0x34b6[1287]] * local$$47186[_0x34b6[1287]] + local$$47186[_0x34b6[1484]];
};
/**
 * @param {?} local$$47223
 * @return {?}
 */
LSJMath[_0x34b6[1703]] = function(local$$47223) {
  return local$$47223[_0x34b6[290]] == 0 && local$$47223[_0x34b6[291]] == 0;
};
/**
 * @param {?} local$$47244
 * @return {?}
 */
LSJMath[_0x34b6[1704]] = function(local$$47244) {
  return local$$47244[_0x34b6[290]] == 0 && local$$47244[_0x34b6[291]] == 0 && local$$47244[_0x34b6[1287]] == 0;
};
/**
 * @param {?} local$$47271
 * @param {?} local$$47272
 * @return {?}
 */
LSJMath[_0x34b6[1705]] = function(local$$47271, local$$47272) {
  return Math[_0x34b6[1525]](local$$47271[_0x34b6[1693]] / LSJMath[_0x34b6[1702]](local$$47271[_0x34b6[658]], local$$47272));
};
/**
 * @param {?} local$$47297
 * @param {?} local$$47298
 * @param {?} local$$47299
 * @return {?}
 */
LSJMath[_0x34b6[1706]] = function(local$$47297, local$$47298, local$$47299) {
  var local$$47304 = local$$47297[_0x34b6[1287]];
  var local$$47309 = local$$47297[_0x34b6[1484]];
  var local$$47314 = local$$47298[_0x34b6[1280]];
  var local$$47319 = local$$47299[_0x34b6[1280]];
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
  var local$$47454 = .7071067811 / Math[_0x34b6[889]](local$$47364[_0x34b6[1707]]() + local$$47409[_0x34b6[1707]]());
  local$$47437[_0x34b6[350]](local$$47454);
  return local$$47437;
};
/**
 * @param {number} local$$47469
 * @return {?}
 */
LSJMath[_0x34b6[1362]] = function(local$$47469) {
  return 0 === (local$$47469 & local$$47469 - 1);
};
/**
 * @param {number} local$$47484
 * @return {?}
 */
LSJMath[_0x34b6[1701]] = function(local$$47484) {
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
LSJMath[_0x34b6[1366]] = function(local$$47514) {
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
  LSJGeometry[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1708];
  this[_0x34b6[430]] = new THREE.Vector3(0, 0, 0);
  this[_0x34b6[1447]] = new THREE.Sphere;
};
LSJGeoModel[_0x34b6[219]] = Object[_0x34b6[242]](LSJGeometry[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJGeoModel[_0x34b6[219]][_0x34b6[1183]] = LSJGeoModel;
/**
 * @return {undefined}
 */
LSJGeoModel[_0x34b6[219]][_0x34b6[1261]] = function() {
};
/**
 * @param {?} local$$47615
 * @param {?} local$$47616
 * @param {?} local$$47617
 * @return {undefined}
 */
LSJGeoModel[_0x34b6[219]][_0x34b6[1241]] = function(local$$47615, local$$47616, local$$47617) {
  this[_0x34b6[430]][_0x34b6[290]] = local$$47615;
  this[_0x34b6[430]][_0x34b6[291]] = local$$47616;
  this[_0x34b6[430]][_0x34b6[1287]] = local$$47617;
};
/**
 * @return {?}
 */
LSJGeoModel[_0x34b6[219]][_0x34b6[1240]] = function() {
  return this[_0x34b6[1690]];
};
/**
 * @param {string} local$$47668
 * @return {undefined}
 */
LSJGeoModel[_0x34b6[219]][_0x34b6[1060]] = function(local$$47668) {
  /**
   * @param {?} local$$47670
   * @return {undefined}
   */
  var local$$47693 = function(local$$47670) {
    if (local$$47670[_0x34b6[1709]]) {
      /** @type {number} */
      var local$$47684 = local$$47670[_0x34b6[1710]] / local$$47670[_0x34b6[1711]] * 100;
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
  var local$$47718 = local$$47703[_0x34b6[388]](0, local$$47703[_0x34b6[223]] - 3) + _0x34b6[1712];
  var local$$47733 = local$$47718[_0x34b6[474]](0, local$$47718[_0x34b6[382]](_0x34b6[1161]) + 1);
  var local$$47750 = local$$47718[_0x34b6[474]](local$$47718[_0x34b6[382]](_0x34b6[1161]) + 1, local$$47718[_0x34b6[223]]);
  var local$$47752 = this;
  var local$$47756 = new THREE.MTLLoader;
  local$$47756[_0x34b6[1059]](local$$47733);
  local$$47756[_0x34b6[1060]](local$$47750, function(local$$47766) {
    local$$47766[_0x34b6[1713]]();
    var local$$47775 = new THREE.OBJLoader;
    local$$47775[_0x34b6[1074]](local$$47766);
    local$$47775[_0x34b6[1060]](local$$47703, function(local$$47785) {
      /** @type {boolean} */
      local$$47785[_0x34b6[1714]] = true;
      /** @type {boolean} */
      local$$47785[_0x34b6[1715]] = true;
      local$$47752[_0x34b6[1446]][_0x34b6[274]](local$$47785);
      /** @type {number} */
      var local$$47808 = 0;
      for (; local$$47808 < local$$47785[_0x34b6[684]][_0x34b6[223]]; local$$47808++) {
        if (local$$47785[_0x34b6[684]][local$$47808][_0x34b6[445]] == _0x34b6[1348]) {
          /** @type {number} */
          var local$$47830 = 0;
          for (; local$$47830 < local$$47785[_0x34b6[684]][local$$47808][_0x34b6[684]][_0x34b6[223]]; local$$47830++) {
            var local$$47856 = local$$47785[_0x34b6[684]][local$$47808][_0x34b6[684]][local$$47830][_0x34b6[1126]];
            /** @type {boolean} */
            local$$47785[_0x34b6[684]][local$$47808][_0x34b6[684]][local$$47830][_0x34b6[1714]] = true;
            /** @type {boolean} */
            local$$47785[_0x34b6[684]][local$$47808][_0x34b6[684]][local$$47830][_0x34b6[1715]] = true;
            if (local$$47856 != undefined && local$$47856[_0x34b6[1447]] === null) {
              local$$47856[_0x34b6[1716]]();
            }
            if (local$$47856 != undefined) {
              local$$47856[_0x34b6[1717]](0);
              LSJMath[_0x34b6[1451]](local$$47752[_0x34b6[1447]], local$$47856[_0x34b6[1447]]);
            }
          }
        } else {
          if (local$$47785[_0x34b6[684]][local$$47808][_0x34b6[445]] == _0x34b6[329]) {
            local$$47856 = local$$47785[_0x34b6[684]][local$$47808][_0x34b6[1126]];
            /** @type {boolean} */
            local$$47785[_0x34b6[684]][local$$47808][_0x34b6[1714]] = true;
            /** @type {boolean} */
            local$$47785[_0x34b6[684]][local$$47808][_0x34b6[1715]] = true;
            if (local$$47856 != undefined && local$$47856[_0x34b6[1447]] === null) {
              local$$47856[_0x34b6[1716]]();
            }
            if (local$$47856 != undefined) {
              local$$47856[_0x34b6[1717]](0);
              LSJMath[_0x34b6[1451]](local$$47752[_0x34b6[1447]], local$$47856[_0x34b6[1447]]);
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
LSJGeoModel[_0x34b6[219]][_0x34b6[1450]] = function() {
  return this[_0x34b6[1447]];
};
/**
 * @return {undefined}
 */
LSJLayer = function() {
  /** @type {!Array} */
  this[_0x34b6[1718]] = [];
  this[_0x34b6[1446]] = new THREE.Group;
  this[_0x34b6[1447]] = new THREE.Sphere;
  /** @type {number} */
  this[_0x34b6[1719]] = 0;
  /** @type {number} */
  getScene()[_0x34b6[1720]] = 0;
  /** @type {!Date} */
  this[_0x34b6[1721]] = new Date;
  getScene()[_0x34b6[1722]] = this[_0x34b6[1721]][_0x34b6[1723]]();
  /** @type {number} */
  this[_0x34b6[1724]] = 0;
};
/** @type {function(): undefined} */
LSJLayer[_0x34b6[219]][_0x34b6[1183]] = LSJLayer;
/**
 * @return {undefined}
 */
LSJLayer[_0x34b6[219]][_0x34b6[232]] = function() {
  var local$$48116 = this[_0x34b6[1718]][_0x34b6[223]];
  /** @type {number} */
  var local$$48119 = 0;
  for (; local$$48119 < local$$48116; local$$48119++) {
    var local$$48128 = this[_0x34b6[1718]][local$$48119];
    /** @type {null} */
    local$$48128[_0x34b6[1446]][_0x34b6[1725]] = null;
    this[_0x34b6[1446]][_0x34b6[1448]](local$$48128[_0x34b6[1446]]);
    if (local$$48128 != null) {
      local$$48128[_0x34b6[232]]();
    }
  }
  this[_0x34b6[1718]][_0x34b6[388]](0, local$$48116);
  /** @type {!Array} */
  this[_0x34b6[1718]] = [];
};
/**
 * @return {undefined}
 */
LSJLayer[_0x34b6[219]][_0x34b6[1726]] = function() {
  var local$$48194 = this[_0x34b6[1718]][_0x34b6[223]];
  /** @type {number} */
  var local$$48197 = 0;
  for (; local$$48197 < local$$48194; local$$48197++) {
    var local$$48206 = this[_0x34b6[1718]][local$$48197];
    /** @type {null} */
    local$$48206[_0x34b6[1446]][_0x34b6[1725]] = null;
    this[_0x34b6[1446]][_0x34b6[1448]](local$$48206[_0x34b6[1446]]);
    if (local$$48206 != null) {
      local$$48206[_0x34b6[232]]();
    }
  }
  this[_0x34b6[1718]][_0x34b6[388]](0, local$$48194);
  /** @type {!Array} */
  this[_0x34b6[1718]] = [];
};
/**
 * @return {?}
 */
LSJLayer[_0x34b6[219]][_0x34b6[1450]] = function() {
  if (this[_0x34b6[1447]][_0x34b6[1456]]()) {
    var local$$48279 = this[_0x34b6[1718]][_0x34b6[223]];
    /** @type {number} */
    var local$$48282 = 0;
    for (; local$$48282 < local$$48279; local$$48282++) {
      var local$$48291 = this[_0x34b6[1718]][local$$48282];
      if (local$$48291 != undefined) {
        if (local$$48291[_0x34b6[445]] == _0x34b6[1708] || local$$48291[_0x34b6[445]] == _0x34b6[1727]) {
          LSJMath[_0x34b6[1451]](this[_0x34b6[1447]], local$$48291[_0x34b6[1450]]());
        }
      }
    }
  }
  return this[_0x34b6[1447]];
};
/**
 * @param {!Object} local$$48342
 * @return {undefined}
 */
LSJLayer[_0x34b6[219]][_0x34b6[1728]] = function(local$$48342) {
  if (local$$48342 == null || local$$48342 == undefined) {
    return;
  }
  this[_0x34b6[1719]]++;
  local$$48342[_0x34b6[332]] = this[_0x34b6[1719]];
  local$$48342[_0x34b6[1729]] = this;
  this[_0x34b6[1718]][_0x34b6[220]](local$$48342);
  if (local$$48342[_0x34b6[1446]] != null) {
    this[_0x34b6[1446]][_0x34b6[274]](local$$48342[_0x34b6[1446]]);
  }
};
/**
 * @param {?} local$$48406
 * @return {undefined}
 */
LSJLayer[_0x34b6[219]][_0x34b6[1730]] = function(local$$48406) {
  var local$$48412 = this[_0x34b6[1731]](local$$48406);
  /** @type {null} */
  local$$48412[_0x34b6[1446]][_0x34b6[1725]] = null;
  this[_0x34b6[1446]][_0x34b6[1448]](local$$48412[_0x34b6[1446]]);
  if (local$$48412 != null) {
    var local$$48443 = this[_0x34b6[1718]][_0x34b6[742]](local$$48412);
    if (local$$48443 !== -1) {
      this[_0x34b6[1718]][_0x34b6[222]](local$$48443, 1);
    }
    local$$48412[_0x34b6[232]]();
    return;
  }
};
/**
 * @param {?} local$$48477
 * @return {undefined}
 */
LSJLayer[_0x34b6[219]][_0x34b6[1732]] = function(local$$48477) {
  var local$$48483 = this[_0x34b6[1733]](local$$48477);
  /** @type {null} */
  local$$48483[_0x34b6[1446]][_0x34b6[1725]] = null;
  this[_0x34b6[1446]][_0x34b6[1448]](local$$48483[_0x34b6[1446]]);
  if (local$$48483 != null) {
    var local$$48514 = this[_0x34b6[1718]][_0x34b6[742]](local$$48483);
    if (local$$48514 !== -1) {
      this[_0x34b6[1718]][_0x34b6[222]](local$$48514, 1);
    }
    local$$48483[_0x34b6[232]]();
    return;
  }
};
/**
 * @param {?} local$$48548
 * @return {?}
 */
LSJLayer[_0x34b6[219]][_0x34b6[1731]] = function(local$$48548) {
  var local$$48556 = this[_0x34b6[1718]][_0x34b6[223]];
  /** @type {number} */
  var local$$48559 = 0;
  for (; local$$48559 < local$$48556; local$$48559++) {
    var local$$48568 = this[_0x34b6[1718]][local$$48559];
    if (local$$48568 != null) {
      if (local$$48568[_0x34b6[1115]] == local$$48548) {
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
LSJLayer[_0x34b6[219]][_0x34b6[1733]] = function(local$$48596) {
  var local$$48604 = this[_0x34b6[1718]][_0x34b6[223]];
  /** @type {number} */
  var local$$48607 = 0;
  for (; local$$48607 < local$$48604; local$$48607++) {
    var local$$48616 = this[_0x34b6[1718]][local$$48607];
    if (local$$48616 != null) {
      if (local$$48616[_0x34b6[332]] == local$$48596) {
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
LSJLayer[_0x34b6[219]][_0x34b6[1734]] = function(local$$48644) {
  var local$$48652 = this[_0x34b6[1718]][_0x34b6[223]];
  if (local$$48644 >= 0 && local$$48644 < local$$48652) {
    return this[_0x34b6[1718]][local$$48644];
  }
  return null;
};
/**
 * @param {?} local$$48678
 * @return {?}
 */
LSJLayer[_0x34b6[219]][_0x34b6[225]] = function(local$$48678) {
  /** @type {!Date} */
  var local$$48681 = new Date;
  getScene()[_0x34b6[1722]] = local$$48681[_0x34b6[1723]]();
  var local$$48699 = this[_0x34b6[1718]][_0x34b6[223]];
  /** @type {number} */
  var local$$48702 = 0;
  /** @type {number} */
  var local$$48705 = 0;
  for (; local$$48705 < local$$48699; local$$48705++) {
    var local$$48714 = this[_0x34b6[1718]][local$$48705];
    if (local$$48714 != null) {
      if (local$$48714[_0x34b6[445]] == _0x34b6[1682] || local$$48714[_0x34b6[445]] == _0x34b6[1290] || local$$48714[_0x34b6[445]] == _0x34b6[1727] || local$$48714[_0x34b6[445]] == _0x34b6[1735]) {
        if (local$$48714[_0x34b6[445]] == _0x34b6[1290]) {
          /** @type {!Date} */
          var local$$48752 = new Date;
          var local$$48758 = local$$48752[_0x34b6[1723]]();
          if (local$$48758 - getScene()[_0x34b6[1722]] > 10 || this[_0x34b6[1724]] > local$$48705) {
            local$$48714[_0x34b6[225]](local$$48678, false);
          } else {
            local$$48714[_0x34b6[225]](local$$48678, true);
            /** @type {number} */
            local$$48702 = local$$48705;
          }
        } else {
          local$$48714[_0x34b6[225]](local$$48678);
          if (getScene()[_0x34b6[1720]] > 2) {
            break;
          }
        }
      }
    }
  }
  /** @type {number} */
  this[_0x34b6[1724]] = local$$48702 == local$$48699 - 1 ? 0 : local$$48702;
  return null;
};
/**
 * @return {undefined}
 */
LSJPageLOD = function() {
  this[_0x34b6[1115]] = _0x34b6[381];
  this[_0x34b6[445]] = _0x34b6[1736];
  /** @type {boolean} */
  this[_0x34b6[330]] = true;
  /** @type {number} */
  this[_0x34b6[1737]] = 0;
  /** @type {number} */
  this[_0x34b6[1738]] = 200;
  this[_0x34b6[1739]] = _0x34b6[381];
  this[_0x34b6[1740]] = LSELoadStatus[_0x34b6[1741]];
  /** @type {!Array} */
  this[_0x34b6[755]] = [];
  /** @type {!Array} */
  this[_0x34b6[1742]] = [];
  this[_0x34b6[1743]] = new THREE.Frustum;
  this[_0x34b6[1744]] = new THREE.Vector4;
  this[_0x34b6[1745]] = new THREE.Matrix4;
  this[_0x34b6[1746]] = new THREE.Matrix4;
  this[_0x34b6[1747]] = new THREE.Matrix4;
  this[_0x34b6[1748]] = new THREE.Matrix4;
  this[_0x34b6[1749]] = new THREE.Vector4;
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {number} */
  this[_0x34b6[1750]] = 0;
  /** @type {number} */
  this[_0x34b6[1751]] = 0;
  /** @type {number} */
  this[_0x34b6[1752]] = 2;
  /** @type {number} */
  this[_0x34b6[1753]] = 0;
  /** @type {number} */
  this[_0x34b6[1754]] = 2;
  /** @type {number} */
  this[_0x34b6[1755]] = 0;
  /** @type {number} */
  this[_0x34b6[1756]] = 2;
  /** @type {number} */
  this[_0x34b6[1757]] = 0;
  /** @type {number} */
  this[_0x34b6[1758]] = 0;
  this[_0x34b6[1759]] = new THREE.Sphere;
};
/** @type {function(): undefined} */
LSJPageLOD[_0x34b6[219]][_0x34b6[1183]] = LSJPageLOD;
/**
 * @param {?} local$$49027
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1760]] = function(local$$49027) {
  this[_0x34b6[755]][_0x34b6[220]](local$$49027);
  this[_0x34b6[1742]][_0x34b6[220]](local$$49027);
  this[_0x34b6[1446]][_0x34b6[274]](local$$49027[_0x34b6[1446]]);
  local$$49027[_0x34b6[1761]] = this;
  local$$49027[_0x34b6[1762]] = local$$49027;
};
/**
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1763]] = function() {
  return this[_0x34b6[1749]];
};
/**
 * @param {?} local$$49091
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1764]] = function(local$$49091) {
  this[_0x34b6[1749]] = local$$49091;
};
/**
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1765]] = function() {
  return this[_0x34b6[1747]];
};
/**
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1766]] = function() {
  return this[_0x34b6[1743]];
};
/**
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1767]] = function() {
  return this[_0x34b6[1744]];
};
/**
 * @param {?} local$$49153
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1768]] = function(local$$49153) {
  this[_0x34b6[1744]] = local$$49153;
};
/**
 * @param {?} local$$49170
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1769]] = function(local$$49170) {
  this[_0x34b6[1751]] = local$$49170;
};
/**
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1770]] = function() {
  return this[_0x34b6[1751]];
};
/**
 * @param {?} local$$49202
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1771]] = function(local$$49202) {
  this[_0x34b6[1750]] = local$$49202;
};
/**
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1772]] = function() {
  return this[_0x34b6[1750]];
};
/**
 * @param {?} local$$49234
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1773]] = function(local$$49234) {
  this[_0x34b6[1737]] -= local$$49234;
};
/**
 * @param {?} local$$49251
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1774]] = function(local$$49251) {
  this[_0x34b6[1737]] += local$$49251;
};
/**
 * @param {?} local$$49268
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[452]] = function(local$$49268) {
  if (local$$49268 == _0x34b6[381]) {
    return;
  }
  var local$$49281 = LSJUtility[_0x34b6[1380]]();
  if (local$$49281 == null) {
    return;
  }
  this[_0x34b6[1739]] = local$$49268;
  var local$$49294 = this;
  this[_0x34b6[1740]] = LSELoadStatus[_0x34b6[1775]];
  /**
   * @return {undefined}
   */
  local$$49281[_0x34b6[1236]] = function() {
    if (local$$49281[_0x34b6[1237]] == 4) {
      if (local$$49281[_0x34b6[1045]] == 200) {
        var local$$49321 = local$$49281[_0x34b6[1776]];
        if (!local$$49321 && local$$49281[_0x34b6[1046]] != _0x34b6[381]) {
          local$$49321 = LSJUtility[_0x34b6[1384]]();
          if (local$$49321 != null) {
            if (window[_0x34b6[1777]]) {
              local$$49321[_0x34b6[1778]](local$$49281[_0x34b6[1046]]);
            } else {
              /** @type {!DOMParser} */
              var local$$49352 = new DOMParser;
              local$$49321 = local$$49352[_0x34b6[935]](local$$49281[_0x34b6[1046]], _0x34b6[1779]);
            }
          }
        }
        if (local$$49321 != null) {
          var local$$49382 = local$$49321[_0x34b6[1781]](_0x34b6[1780])[0];
          if (local$$49382 && local$$49382[_0x34b6[409]]) {
            var local$$49394 = local$$49382[_0x34b6[409]][_0x34b6[406]];
            var local$$49402 = local$$49394[_0x34b6[379]](_0x34b6[477]);
            if (local$$49402[_0x34b6[223]] > 2) {
            }
          }
          local$$49382 = local$$49321[_0x34b6[1781]](_0x34b6[1782])[0];
          if (local$$49382 && local$$49382[_0x34b6[409]]) {
            local$$49394 = local$$49382[_0x34b6[409]][_0x34b6[406]];
            local$$49402 = local$$49394[_0x34b6[379]](_0x34b6[477]);
            if (local$$49402[_0x34b6[223]] > 2) {
            }
          }
          local$$49382 = local$$49321[_0x34b6[1781]](_0x34b6[1783])[0];
          if (local$$49382 && local$$49382[_0x34b6[409]]) {
            local$$49394 = local$$49382[_0x34b6[409]][_0x34b6[406]];
            local$$49402 = local$$49394[_0x34b6[379]](_0x34b6[477]);
            if (local$$49402[_0x34b6[223]] > 2) {
            }
          }
          var local$$49502 = local$$49321[_0x34b6[1781]](_0x34b6[1784])[0];
          var local$$49510 = local$$49502[_0x34b6[684]][_0x34b6[223]];
          /** @type {number} */
          var local$$49513 = 0;
          for (; local$$49513 < local$$49510; local$$49513++) {
            var local$$49522 = local$$49502[_0x34b6[684]][local$$49513];
            /** @type {null} */
            local$$49394 = null;
            if (local$$49522[_0x34b6[409]] != null) {
              local$$49394 = local$$49522[_0x34b6[409]][_0x34b6[406]];
            } else {
              local$$49394 = local$$49522[_0x34b6[785]];
            }
            if (local$$49394 != _0x34b6[381]) {
              var local$$49554 = new LSJPageLODNode;
              local$$49554[_0x34b6[1785]] = LSJUtility[_0x34b6[1373]](LSJUtility[_0x34b6[1372]](local$$49268), local$$49394);
              local$$49294[_0x34b6[1760]](local$$49554);
            }
          }
        }
      }
      this[_0x34b6[1740]] = LSELoadStatus[_0x34b6[1786]];
    }
  };
  local$$49281[_0x34b6[452]](_0x34b6[1044], local$$49268, true);
  local$$49281[_0x34b6[1049]]();
};
/**
 * @param {?} local$$49620
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1787]] = function(local$$49620) {
  if (local$$49620 == _0x34b6[381]) {
    return;
  }
  this[_0x34b6[1739]] = local$$49620;
  var local$$49634 = this;
  this[_0x34b6[1740]] = LSELoadStatus[_0x34b6[1775]];
  var local$$49646 = new THREE.XHRLoader;
  local$$49646[_0x34b6[1060]](local$$49620, function(local$$49651) {
    var local$$49658 = JSON[_0x34b6[768]](local$$49651);
    var local$$49663 = local$$49658[_0x34b6[1788]];
    if (local$$49663 !== undefined) {
      if (local$$49663[_0x34b6[1789]] !== undefined) {
        var local$$49687 = new THREE.Vector3(local$$49663[_0x34b6[1789]].West, local$$49663[_0x34b6[1789]].South, local$$49663[_0x34b6[1789]].MinZ);
        var local$$49706 = new THREE.Vector3(local$$49663[_0x34b6[1789]].East, local$$49663[_0x34b6[1789]].North, local$$49663[_0x34b6[1789]].MaxZ);
        var local$$49710 = new THREE.Vector3;
        local$$49710[_0x34b6[334]](local$$49687[_0x34b6[290]] / 2 + local$$49706[_0x34b6[290]] / 2, local$$49687[_0x34b6[291]] / 2 + local$$49706[_0x34b6[291]] / 2, local$$49687[_0x34b6[1287]] / 2 + local$$49706[_0x34b6[1287]] / 2);
        var local$$49752 = new THREE.Vector3;
        local$$49752[_0x34b6[1697]](local$$49706, local$$49687);
        local$$49634[_0x34b6[1759]][_0x34b6[334]](local$$49710, local$$49752[_0x34b6[223]]() / 2);
      }
      var local$$49785 = local$$49663[_0x34b6[1784]][_0x34b6[1790]][_0x34b6[223]];
      /** @type {number} */
      var local$$49788 = 0;
      for (; local$$49788 < local$$49785; local$$49788++) {
        var local$$49800 = local$$49663[_0x34b6[1784]][_0x34b6[1790]][local$$49788];
        if (local$$49800 != _0x34b6[381]) {
          var local$$49806 = new LSJPageLODNode;
          local$$49806[_0x34b6[1785]] = LSJUtility[_0x34b6[1373]](LSJUtility[_0x34b6[1372]](local$$49620), local$$49800);
          local$$49634[_0x34b6[1760]](local$$49806);
        }
      }
      this[_0x34b6[1740]] = LSELoadStatus[_0x34b6[1786]];
    }
  });
};
/**
 * @param {?} local$$49855
 * @param {?} local$$49856
 * @param {!NodeList} local$$49857
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1791]] = function(local$$49855, local$$49856, local$$49857) {
  if (local$$49856[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
    return;
  }
  /** @type {number} */
  var local$$49872 = 0;
  var local$$49880 = local$$49856[_0x34b6[684]][_0x34b6[223]];
  for (; local$$49872 < local$$49880; local$$49872++) {
    this[_0x34b6[1791]](local$$49855, local$$49856[_0x34b6[684]][local$$49872], local$$49857);
  }
  if (local$$49856 == local$$49856[_0x34b6[1762]]) {
    return;
  }
  if (local$$49856[_0x34b6[1785]] == _0x34b6[381]) {
    return;
  }
  /** @type {number} */
  var local$$49916 = 0;
  var local$$49921 = local$$49857[_0x34b6[223]];
  for (; local$$49916 < local$$49921; local$$49916++) {
    var local$$49927 = local$$49857[local$$49916];
    var local$$49929;
    var local$$49931;
    local$$49929 = local$$49927[_0x34b6[1793]]();
    local$$49931 = local$$49856[_0x34b6[1793]]();
    if (local$$49931 > local$$49929) {
      local$$49857[_0x34b6[222]](local$$49916, 0, local$$49856);
      return;
    } else {
      if (local$$49931 == local$$49929) {
        if (local$$49927[_0x34b6[667]] == local$$49856[_0x34b6[667]]) {
          var local$$49965 = local$$49927[_0x34b6[667]];
          if (local$$49965 != null) {
            /** @type {number} */
            var local$$49970 = -1;
            /** @type {number} */
            var local$$49973 = -1;
            /** @type {number} */
            var local$$49976 = 0;
            for (; local$$49976 < local$$49965[_0x34b6[684]][_0x34b6[223]]; local$$49976++) {
              if (local$$49965[_0x34b6[684]][local$$49872] == local$$49927) {
                /** @type {number} */
                local$$49970 = local$$49872;
              }
              if (local$$49965[_0x34b6[684]][local$$49872] == local$$49856) {
                /** @type {number} */
                local$$49973 = local$$49872;
              }
              if (local$$49970 > -1 && local$$49973 > -1) {
                break;
              }
            }
            if (local$$49973 > local$$49970) {
              local$$49857[_0x34b6[222]](local$$49916, 0, local$$49856);
              return;
            }
          }
        }
      }
    }
  }
  local$$49857[_0x34b6[220]](local$$49856);
};
/**
 * @param {?} local$$50054
 * @param {?} local$$50055
 * @param {?} local$$50056
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1794]] = function(local$$50054, local$$50055, local$$50056) {
  if (local$$50055[_0x34b6[1772]]() < this[_0x34b6[1772]]()) {
    this[_0x34b6[1791]](local$$50054, local$$50055, local$$50056);
  } else {
    /** @type {number} */
    var local$$50074 = 0;
    var local$$50082 = local$$50055[_0x34b6[684]][_0x34b6[223]];
    for (; local$$50074 < local$$50082; local$$50074++) {
      this[_0x34b6[1794]](local$$50054, local$$50055[_0x34b6[684]][local$$50074], local$$50056);
    }
  }
};
/**
 * @param {!Object} local$$50110
 * @param {?} local$$50111
 * @return {?}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1795]] = function(local$$50110, local$$50111) {
  if (gdMemUsed < gdMaxMemAllowed) {
    return false;
  }
  if (local$$50110 != null) {
    if (local$$50110[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
      return false;
    }
    /** @type {number} */
    var local$$50137 = 0;
    var local$$50145 = local$$50110[_0x34b6[684]][_0x34b6[223]];
    for (; local$$50137 < local$$50145; local$$50137++) {
      this[_0x34b6[1795]](local$$50110[_0x34b6[684]][local$$50137], local$$50111);
    }
    if (gdMemUsed < gdMaxMemAllowed) {
      return false;
    }
    if (!local$$50111 && local$$50110 == local$$50110[_0x34b6[1762]]) {
      return false;
    }
    if (local$$50110[_0x34b6[1785]] == _0x34b6[381]) {
      return false;
    }
    /** @type {number} */
    var local$$50199 = this[_0x34b6[1772]]() - local$$50110[_0x34b6[1772]]();
    /** @type {number} */
    var local$$50210 = this[_0x34b6[1770]]() - local$$50110[_0x34b6[1770]]();
    if (local$$50199 < 1) {
      return false;
    }
    if (local$$50110[_0x34b6[1796]]()) {
      local$$50110[_0x34b6[1797]]();
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
LSJPageLOD[_0x34b6[219]][_0x34b6[1798]] = function(local$$50250) {
  local$$50250[_0x34b6[1263]]();
  var local$$50259 = new THREE.Matrix4;
  local$$50259[_0x34b6[1491]](local$$50250[_0x34b6[1285]]);
  this[_0x34b6[1747]][_0x34b6[1286]](local$$50259, this[_0x34b6[1745]]);
  this[_0x34b6[1748]][_0x34b6[1286]](local$$50250[_0x34b6[335]], this[_0x34b6[1747]]);
  this[_0x34b6[1743]][_0x34b6[1492]](this[_0x34b6[1748]]);
  this[_0x34b6[1749]] = LSJMath[_0x34b6[1706]](this[_0x34b6[1744]], local$$50250[_0x34b6[335]], this[_0x34b6[1747]]);
};
/**
 * @param {?} local$$50328
 * @param {?} local$$50329
 * @return {?}
 */
function nodeDistCompare(local$$50328, local$$50329) {
  return local$$50328[_0x34b6[1799]] - local$$50329[_0x34b6[1799]];
}
/**
 * @param {?} local$$50347
 * @return {undefined}
 */
LSJPageLOD[_0x34b6[219]][_0x34b6[1261]] = function(local$$50347) {
  var local$$50352 = this[_0x34b6[755]];
  this[_0x34b6[1751]] = (new Date)[_0x34b6[1723]]();
  ++this[_0x34b6[1750]];
  this[_0x34b6[1798]](local$$50347);
  /** @type {number} */
  this[_0x34b6[1758]] = 0;
  /** @type {number} */
  var local$$50381 = 0;
  var local$$50386 = local$$50352[_0x34b6[223]];
  for (; local$$50381 < local$$50386; local$$50381++) {
    local$$50352[local$$50381][_0x34b6[1261]](local$$50347);
  }
  if (gdMemUsed > gdMaxMemAllowed) {
    /** @type {number} */
    var local$$50402 = 0;
    var local$$50407 = local$$50352[_0x34b6[223]];
    for (; local$$50402 < local$$50407; local$$50402++) {
      this[_0x34b6[1795]](this[_0x34b6[755]][local$$50402], false);
    }
  }
};
/**
 * @return {undefined}
 */
LSJNodeMaterial = function() {
  /** @type {number} */
  this[_0x34b6[332]] = -1;
  this[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1741]];
  this[_0x34b6[1800]] = _0x34b6[381];
  /** @type {null} */
  this[_0x34b6[268]] = null;
  /** @type {boolean} */
  this[_0x34b6[1801]] = false;
};
/**
 * @return {undefined}
 */
LSJPageLODNode = function() {
  this[_0x34b6[445]] = _0x34b6[1802];
  /** @type {!Array} */
  this[_0x34b6[684]] = [];
  /** @type {!Array} */
  this[_0x34b6[1803]] = [];
  /** @type {null} */
  this[_0x34b6[1761]] = null;
  /** @type {null} */
  this[_0x34b6[667]] = null;
  /** @type {null} */
  this[_0x34b6[1762]] = null;
  this[_0x34b6[1785]] = _0x34b6[381];
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {boolean} */
  this[_0x34b6[1804]] = false;
  /** @type {boolean} */
  this[_0x34b6[1805]] = false;
  this[_0x34b6[1759]] = new THREE.Sphere;
  this[_0x34b6[1806]] = new THREE.Box3;
  this[_0x34b6[1807]] = LSELoadStatus[_0x34b6[1741]];
  /** @type {number} */
  this[_0x34b6[1808]] = 0;
  /** @type {number} */
  this[_0x34b6[1750]] = 0;
  /** @type {number} */
  this[_0x34b6[1751]] = 0;
  /** @type {boolean} */
  this[_0x34b6[1809]] = false;
  /** @type {!Array} */
  this[_0x34b6[1810]] = [];
  /** @type {!Array} */
  this[_0x34b6[1811]] = [];
  /** @type {null} */
  this[_0x34b6[1812]] = null;
  /** @type {number} */
  this[_0x34b6[1799]] = 0;
  /** @type {number} */
  this[_0x34b6[1813]] = 0;
};
/** @type {function(): undefined} */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1183]] = LSJPageLODNode;
/**
 * @param {?} local$$50629
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1814]] = function(local$$50629) {
  this[_0x34b6[1805]] = local$$50629;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1815]] = function() {
  return this[_0x34b6[1805]];
};
/**
 * @param {?} local$$50661
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1816]] = function(local$$50661) {
  this[_0x34b6[1807]] = local$$50661;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1817]] = function() {
  return this[_0x34b6[1809]];
};
/**
 * @param {?} local$$50693
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1818]] = function(local$$50693) {
  this[_0x34b6[1809]] = local$$50693;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1792]] = function() {
  return this[_0x34b6[1807]];
};
/**
 * @param {?} local$$50725
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1769]] = function(local$$50725) {
  this[_0x34b6[1751]] = local$$50725;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1770]] = function() {
  return this[_0x34b6[1751]];
};
/**
 * @param {?} local$$50757
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1771]] = function(local$$50757) {
  this[_0x34b6[1750]] = local$$50757;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1772]] = function() {
  return this[_0x34b6[1750]];
};
/**
 * @param {?} local$$50789
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1760]] = function(local$$50789) {
  this[_0x34b6[684]][_0x34b6[220]](local$$50789);
  local$$50789[_0x34b6[1761]] = this[_0x34b6[1761]];
  local$$50789[_0x34b6[1762]] = this[_0x34b6[1762]];
  local$$50789[_0x34b6[667]] = this;
  this[_0x34b6[1446]][_0x34b6[274]](local$$50789[_0x34b6[1446]]);
};
/**
 * @param {?} local$$50841
 * @param {?} local$$50842
 * @param {?} local$$50843
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1089]] = function(local$$50841, local$$50842, local$$50843) {
  if (local$$50842[_0x34b6[1755]] > local$$50842[_0x34b6[1754]]) {
    return;
  }
  local$$50841[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1775]];
  local$$50842[_0x34b6[1755]]++;
  var local$$50871 = new THREE.Texture;
  var local$$50879 = document[_0x34b6[424]](_0x34b6[559]);
  local$$50879[_0x34b6[551]] = local$$50841[_0x34b6[1800]];
  var local$$50889;
  /**
   * @param {?} local$$50894
   * @param {?} local$$50895
   * @return {undefined}
   */
  local$$50879[_0x34b6[556]] = function(local$$50894, local$$50895) {
    if (local$$50841[_0x34b6[1801]]) {
      window[_0x34b6[579]][_0x34b6[1819]](local$$50841[_0x34b6[1800]]);
    }
    local$$50841[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    local$$50842[_0x34b6[1755]]--;
  };
  /**
   * @param {?} local$$50934
   * @param {?} local$$50935
   * @return {undefined}
   */
  local$$50879[_0x34b6[443]] = function(local$$50934, local$$50935) {
    window[_0x34b6[579]][_0x34b6[1819]](local$$50879[_0x34b6[551]]);
    local$$50871[_0x34b6[554]] = local$$50879;
    /** @type {boolean} */
    local$$50871[_0x34b6[1275]] = true;
    local$$50871[_0x34b6[294]] = THREE[_0x34b6[1081]];
    local$$50871[_0x34b6[1092]] = THREE[_0x34b6[1083]];
    local$$50871[_0x34b6[1093]] = THREE[_0x34b6[1083]];
    local$$50871[_0x34b6[1820]] = THREE[_0x34b6[205]];
    local$$50871[_0x34b6[1821]] = THREE[_0x34b6[205]];
    /** @type {boolean} */
    local$$50871[_0x34b6[297]] = false;
    /** @type {number} */
    var local$$51006 = 3;
    if (local$$50871[_0x34b6[1194]] == THREE[_0x34b6[206]]) {
      /** @type {number} */
      local$$51006 = 4;
    }
    /** @type {number} */
    var local$$51029 = local$$50879[_0x34b6[208]] * local$$50879[_0x34b6[209]] * local$$51006;
    gdMemUsed = gdMemUsed + local$$51029;
    local$$50841[_0x34b6[268]][_0x34b6[645]] = local$$50871;
    /** @type {boolean} */
    local$$50841[_0x34b6[268]][_0x34b6[1275]] = true;
    if (local$$50841[_0x34b6[1801]]) {
      window[_0x34b6[579]][_0x34b6[1819]](local$$50841[_0x34b6[1800]]);
    }
    local$$50841[_0x34b6[1800]] = _0x34b6[381];
    local$$50841[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    local$$50842[_0x34b6[1755]]--;
  };
  return local$$50871;
};
/**
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1822]] = function() {
  if (this[_0x34b6[1761]][_0x34b6[1753]] > this[_0x34b6[1761]][_0x34b6[1752]]) {
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
    local$$51129[_0x34b6[1816]](LSELoadStatus.LS_NET_LOADED);
    local$$51129[_0x34b6[1761]][_0x34b6[1753]]--;
  };
  this[_0x34b6[1816]](LSELoadStatus.LS_NET_LOADING);
  /** @type {!XMLHttpRequest} */
  var local$$51155 = new XMLHttpRequest;
  local$$51155[_0x34b6[452]](_0x34b6[1044], this[_0x34b6[1785]], true);
  local$$51155[_0x34b6[1823]] = _0x34b6[1824];
  this[_0x34b6[1761]][_0x34b6[1753]]++;
  local$$51155[_0x34b6[1049]](null);
  var local$$51129 = this;
  /**
   * @return {undefined}
   */
  local$$51155[_0x34b6[1236]] = function() {
    if (local$$51155[_0x34b6[1237]] == 4) {
      if (local$$51155[_0x34b6[1045]] == 200) {
        local$$51129[_0x34b6[1812]] = local$$51155[_0x34b6[1825]];
      } else {
      }
      local$$51129[_0x34b6[1816]](LSELoadStatus.LS_NET_LOADED);
      local$$51129[_0x34b6[1761]][_0x34b6[1753]]--;
    }
  };
};
/**
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1060]] = function() {
  if (this[_0x34b6[1761]][_0x34b6[1757]] > this[_0x34b6[1761]][_0x34b6[1756]]) {
    return;
  }
  if (this[_0x34b6[1812]] == null) {
    this[_0x34b6[1816]](LSELoadStatus.LS_LOADED);
    return;
  }
  this[_0x34b6[1816]](LSELoadStatus.LS_LOADING);
  var local$$51285 = this;
  this[_0x34b6[1761]][_0x34b6[1757]]++;
  /** @type {!Worker} */
  var local$$51299 = new Worker(_0x34b6[1826]);
  /**
   * @param {?} local$$51304
   * @return {undefined}
   */
  local$$51299[_0x34b6[1827]] = function(local$$51304) {
    var local$$51309 = local$$51304[_0x34b6[575]];
    if (local$$51309 != null && local$$51309 != undefined) {
      var local$$51315;
      {
        var local$$51320 = local$$51285[_0x34b6[1785]];
        local$$51315 = local$$51320[_0x34b6[702]](0, local$$51320[_0x34b6[382]](_0x34b6[1161]) + 1);
      }
      var local$$51344 = local$$51309[_0x34b6[1810]][_0x34b6[223]];
      /** @type {number} */
      var local$$51347 = 0;
      for (; local$$51347 < local$$51344; local$$51347++) {
        var local$$51356 = local$$51309[_0x34b6[1810]][local$$51347];
        var local$$51359 = new LSJNodeMaterial;
        if (local$$51356[_0x34b6[1800]] != _0x34b6[381]) {
          if (local$$51356[_0x34b6[1828]]) {
            local$$51359[_0x34b6[1800]] = local$$51315 + local$$51356[_0x34b6[1800]];
          } else {
            local$$51359[_0x34b6[1800]] = local$$51356[_0x34b6[1800]];
          }
        } else {
          if (local$$51356[_0x34b6[1829]] != null) {
            local$$51359[_0x34b6[1800]] = window[_0x34b6[579]][_0x34b6[1830]](local$$51356[_0x34b6[1829]]);
            /** @type {null} */
            local$$51356[_0x34b6[1829]] = null;
            /** @type {boolean} */
            local$$51359[_0x34b6[1801]] = true;
          }
        }
        local$$51359[_0x34b6[268]] = new THREE.MeshBasicMaterial;
        local$$51359[_0x34b6[268]][_0x34b6[245]] = (new THREE.Color)[_0x34b6[1533]](local$$51356[_0x34b6[1831]], local$$51356[_0x34b6[1832]], local$$51356[_0x34b6[1833]]);
        if (local$$51359[_0x34b6[1800]] == _0x34b6[381]) {
          local$$51359[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
        }
        local$$51285[_0x34b6[1810]][_0x34b6[220]](local$$51359);
      }
      local$$51285[_0x34b6[768]](local$$51309, local$$51285[_0x34b6[1810]], local$$51315);
    }
    /** @type {null} */
    local$$51309 = null;
    /** @type {null} */
    local$$51304[_0x34b6[575]] = null;
    /** @type {null} */
    local$$51285[_0x34b6[1812]] = null;
    local$$51285[_0x34b6[1816]](LSELoadStatus.LS_LOADED);
    local$$51285[_0x34b6[1761]][_0x34b6[1757]]--;
  };
  /**
   * @param {?} local$$51533
   * @return {undefined}
   */
  local$$51299[_0x34b6[556]] = function(local$$51533) {
    console[_0x34b6[514]](_0x34b6[1834] + local$$51533[_0x34b6[852]]);
    /** @type {null} */
    local$$51285[_0x34b6[1812]] = null;
    local$$51285[_0x34b6[1816]](LSELoadStatus.LS_LOADED);
    local$$51285[_0x34b6[1761]][_0x34b6[1757]]--;
  };
  local$$51299[_0x34b6[1835]](this[_0x34b6[1812]]);
};
/**
 * @param {!Array} local$$51588
 * @param {?} local$$51589
 * @param {?} local$$51590
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[768]] = function(local$$51588, local$$51589, local$$51590) {
  if (local$$51588 == null || local$$51588 === undefined) {
    return;
  }
  /** @type {number} */
  var local$$51601 = 0;
  var local$$51609 = local$$51588[_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$51601 = 0;
  for (; local$$51601 < local$$51609; local$$51601++) {
    var local$$51618 = new LSJPageLODNode;
    this[_0x34b6[1760]](local$$51618);
    local$$51618[_0x34b6[768]](local$$51588[_0x34b6[684]][local$$51601], local$$51589, local$$51590);
  }
  this[_0x34b6[1808]] = local$$51588[_0x34b6[1808]];
  if (local$$51588[_0x34b6[1803]][_0x34b6[223]] > 0) {
    /** @type {number} */
    local$$51609 = local$$51588[_0x34b6[1803]][_0x34b6[223]] / 2;
    /** @type {number} */
    local$$51601 = 0;
    for (; local$$51601 < local$$51609; local$$51601++) {
      var local$$51671 = new THREE.Vector2;
      local$$51671[_0x34b6[290]] = local$$51588[_0x34b6[1803]][2 * local$$51601];
      local$$51671[_0x34b6[291]] = local$$51588[_0x34b6[1803]][2 * local$$51601 + 1];
      this[_0x34b6[1803]][_0x34b6[220]](local$$51671);
    }
  }
  if (this[_0x34b6[1785]] == _0x34b6[381]) {
    if (local$$51588[_0x34b6[1785]] != _0x34b6[381]) {
      this[_0x34b6[1785]] = local$$51590 + local$$51588[_0x34b6[1785]];
    }
  }
  if (local$$51588[_0x34b6[1759]][_0x34b6[223]] > 0) {
    this[_0x34b6[1759]] = new THREE.Sphere;
    var local$$51753 = new THREE.Vector3;
    local$$51753[_0x34b6[334]](local$$51588[_0x34b6[1759]][0], local$$51588[_0x34b6[1759]][1], local$$51588[_0x34b6[1759]][2]);
    this[_0x34b6[1759]][_0x34b6[334]](local$$51753, local$$51588[_0x34b6[1759]][3]);
    LSJMath[_0x34b6[1451]](this[_0x34b6[1761]][_0x34b6[1759]], this[_0x34b6[1759]]);
  }
  /** @type {number} */
  this[_0x34b6[1813]] = 0;
  var local$$51817 = local$$51588[_0x34b6[1836]][_0x34b6[223]];
  /** @type {number} */
  var local$$51820 = 0;
  for (; local$$51820 < local$$51817; local$$51820++) {
    var local$$51829 = local$$51588[_0x34b6[1836]][local$$51820];
    if (local$$51829[_0x34b6[1837]] != null && local$$51829[_0x34b6[1838]] >= 0 && local$$51829[_0x34b6[1838]] < local$$51589[_0x34b6[223]]) {
      var local$$51852 = new THREE.BufferGeometry;
      if (local$$51829[_0x34b6[1839]] != null) {
        local$$51852[_0x34b6[1355]](new THREE.BufferAttribute(local$$51829[_0x34b6[1839]], 1));
        this[_0x34b6[1813]] += local$$51829[_0x34b6[1839]][_0x34b6[223]] * 2;
      }
      if (local$$51829[_0x34b6[1837]] != null) {
        local$$51852[_0x34b6[1174]](_0x34b6[430], new THREE.BufferAttribute(local$$51829[_0x34b6[1837]], 3));
        this[_0x34b6[1813]] += local$$51829[_0x34b6[1837]][_0x34b6[223]] * 4;
      }
      if (local$$51829[_0x34b6[1129]] != null) {
        local$$51852[_0x34b6[1174]](_0x34b6[570], new THREE.BufferAttribute(local$$51829[_0x34b6[1129]], 3));
        this[_0x34b6[1813]] += local$$51829[_0x34b6[1129]][_0x34b6[223]] * 4;
      }
      if (local$$51829[_0x34b6[674]] != null) {
        local$$51852[_0x34b6[1174]](_0x34b6[245], new THREE.BufferAttribute(local$$51829[_0x34b6[674]], local$$51829[_0x34b6[1840]]));
        this[_0x34b6[1813]] += local$$51829[_0x34b6[674]][_0x34b6[223]] * 4;
      }
      var local$$51996 = local$$51829[_0x34b6[1130]][_0x34b6[223]];
      /** @type {number} */
      k = 0;
      for (; k < local$$51996; k++) {
        if (local$$51829[_0x34b6[1130]][k] != null && local$$51829[_0x34b6[1130]][k] != undefined) {
          local$$51852[_0x34b6[1174]](_0x34b6[1176], new THREE.BufferAttribute(local$$51829[_0x34b6[1130]][k], 2));
          this[_0x34b6[1813]] += local$$51829[_0x34b6[1130]][k][_0x34b6[223]] * 4;
        }
      }
      var local$$52054 = local$$51589[local$$51829[_0x34b6[1838]]];
      var local$$52061 = new THREE.Mesh(local$$51852, local$$52054[_0x34b6[268]]);
      this[_0x34b6[1811]][_0x34b6[220]](local$$52054);
      this[_0x34b6[1446]][_0x34b6[274]](local$$52061);
      this[_0x34b6[1818]](true);
      this[_0x34b6[1761]][_0x34b6[1774]](1);
    }
  }
  gdMemUsed = gdMemUsed + this[_0x34b6[1813]];
  if (this[_0x34b6[1785]] == _0x34b6[381]) {
    this[_0x34b6[1807]] = LSELoadStatus[_0x34b6[1786]];
  }
};
/**
 * @param {?} local$$52131
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1841]] = function(local$$52131) {
  this[_0x34b6[1814]](false);
  var local$$52146 = this[_0x34b6[1761]][_0x34b6[1766]]();
  if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
    if (!local$$52146[_0x34b6[1842]](this[_0x34b6[1759]])) {
      return false;
    }
  } else {
    if (!this[_0x34b6[1806]][_0x34b6[1456]]()) {
      if (!local$$52146[_0x34b6[1843]](this[_0x34b6[1806]])) {
        return false;
      }
    }
  }
  this[_0x34b6[1814]](true);
  return true;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1793]] = function() {
  /** @type {number} */
  var local$$52214 = 0;
  var local$$52219 = this[_0x34b6[667]];
  for (; local$$52219 != null;) {
    local$$52214++;
    local$$52219 = local$$52219[_0x34b6[667]];
  }
  return local$$52214;
};
/**
 * @param {?} local$$52246
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1844]] = function(local$$52246) {
  /** @type {number} */
  var local$$52249 = 0;
  if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
    local$$52249 = LSJMath[_0x34b6[1699]](this[_0x34b6[1759]][_0x34b6[658]], local$$52246[_0x34b6[430]]);
    var local$$52291 = LSJMath[_0x34b6[1700]](this[_0x34b6[1759]][_0x34b6[658]], this[_0x34b6[1761]][_0x34b6[1765]]());
    /** @type {number} */
    var local$$52294 = local$$52291 * local$$52291;
    return local$$52249;
  } else {
    if (!this[_0x34b6[1806]][_0x34b6[1456]]()) {
      local$$52294 = LSJMath[_0x34b6[1700]](this[_0x34b6[1806]][_0x34b6[658]], this[_0x34b6[1761]][_0x34b6[1765]]());
      local$$52249 = LSJMath[_0x34b6[1699]](this[_0x34b6[1806]][_0x34b6[658]], local$$52246[_0x34b6[430]]);
      return local$$52249;
    }
  }
  /** @type {number} */
  var local$$52347 = 0;
  var local$$52355 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$52347 < local$$52355; local$$52347++) {
    var local$$52364 = this[_0x34b6[684]][local$$52347];
    if (local$$52364 != null) {
      local$$52249 = local$$52364[_0x34b6[1844]](local$$52246);
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
LSJPageLODNode[_0x34b6[219]][_0x34b6[1796]] = function() {
  if (this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1741]] && this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1845]] && this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
    return false;
  }
  if (this[_0x34b6[1846]]()) {
    return false;
  }
  /** @type {number} */
  var local$$52438 = 0;
  var local$$52446 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$52438 < local$$52446; local$$52438++) {
    if (!this[_0x34b6[684]][local$$52438][_0x34b6[1796]]()) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1847]] = function() {
  /** @type {number} */
  var local$$52481 = 0;
  var local$$52489 = this[_0x34b6[1811]][_0x34b6[223]];
  for (; local$$52481 < local$$52489; local$$52481++) {
    if (this[_0x34b6[1811]][local$$52481][_0x34b6[1045]] != LSELoadStatus[_0x34b6[1786]]) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1846]] = function() {
  /** @type {number} */
  var local$$52526 = 0;
  var local$$52534 = this[_0x34b6[1811]][_0x34b6[223]];
  for (; local$$52526 < local$$52534; local$$52526++) {
    if (this[_0x34b6[1811]][local$$52526][_0x34b6[1045]] != LSELoadStatus[_0x34b6[1741]] && this[_0x34b6[1811]][local$$52526][_0x34b6[1045]] != LSELoadStatus[_0x34b6[1786]]) {
      return true;
    }
  }
  return false;
};
/**
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1848]] = function() {
  /** @type {number} */
  var local$$52583 = 0;
  if (this[_0x34b6[1817]]()) {
    /** @type {number} */
    local$$52583 = local$$52583 + 1;
  }
  /** @type {number} */
  var local$$52597 = 0;
  var local$$52605 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$52597 < local$$52605; local$$52597++) {
    local$$52583 = local$$52583 + this[_0x34b6[684]][local$$52597][_0x34b6[1848]]();
  }
  return local$$52583;
};
/**
 * @return {undefined}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1797]] = function() {
  /** @type {number} */
  var local$$52637 = 0;
  /** @type {number} */
  var local$$52640 = 0;
  var local$$52648 = this[_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$52640 = 0;
  for (; local$$52640 < local$$52648; local$$52640++) {
    this[_0x34b6[684]][local$$52640][_0x34b6[1797]]();
  }
  this[_0x34b6[684]][_0x34b6[222]](0, local$$52648);
  this[_0x34b6[1803]][_0x34b6[222]](0, this[_0x34b6[1803]][_0x34b6[223]]);
  this[_0x34b6[1811]][_0x34b6[222]](0, this[_0x34b6[1811]][_0x34b6[223]]);
  for (; this[_0x34b6[1810]][_0x34b6[223]] > 0;) {
    var local$$52724 = this[_0x34b6[1810]][_0x34b6[1849]]();
    if (local$$52724[_0x34b6[268]] != null && local$$52724[_0x34b6[268]] != undefined) {
      var local$$52742 = local$$52724[_0x34b6[268]][_0x34b6[645]];
      if (local$$52742 != null && local$$52742 != undefined) {
        if (local$$52742[_0x34b6[554]] != null) {
          /** @type {number} */
          var local$$52754 = 3;
          if (local$$52742[_0x34b6[1194]] == THREE[_0x34b6[206]]) {
            /** @type {number} */
            local$$52754 = 4;
          }
          /** @type {number} */
          var local$$52783 = local$$52742[_0x34b6[554]][_0x34b6[208]] * local$$52742[_0x34b6[554]][_0x34b6[209]] * local$$52754;
          /** @type {number} */
          gdMemUsed = gdMemUsed - local$$52783;
          /** @type {null} */
          local$$52742[_0x34b6[554]] = null;
        }
        local$$52742[_0x34b6[232]]();
      }
      local$$52724[_0x34b6[268]][_0x34b6[232]]();
      /** @type {null} */
      local$$52724[_0x34b6[268]][_0x34b6[645]] = null;
      /** @type {null} */
      local$$52724[_0x34b6[268]] = null;
    }
  }
  /** @type {number} */
  var local$$52844 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]] - 1;
  for (; local$$52844 >= 0; local$$52844--) {
    var local$$52857 = this[_0x34b6[1446]][_0x34b6[684]][local$$52844];
    this[_0x34b6[1446]][_0x34b6[1448]](local$$52857);
    if (local$$52857 != null && local$$52857 instanceof THREE[_0x34b6[329]]) {
      if (local$$52857[_0x34b6[1126]]) {
        local$$52857[_0x34b6[1126]][_0x34b6[232]]();
      }
      if (local$$52857[_0x34b6[268]] != null && local$$52857[_0x34b6[268]] != undefined) {
        if (local$$52857[_0x34b6[268]][_0x34b6[645]] != null && local$$52857[_0x34b6[268]][_0x34b6[645]] != undefined) {
          local$$52857[_0x34b6[268]][_0x34b6[645]][_0x34b6[232]]();
        }
        local$$52857[_0x34b6[268]][_0x34b6[232]]();
      }
      /** @type {null} */
      local$$52857[_0x34b6[268]] = null;
      /** @type {null} */
      local$$52857[_0x34b6[1126]] = null;
      this[_0x34b6[1761]][_0x34b6[1773]](1);
    }
    /** @type {null} */
    local$$52857 = null;
  }
  /** @type {number} */
  gdMemUsed = gdMemUsed - this[_0x34b6[1813]];
  /** @type {number} */
  this[_0x34b6[1813]] = 0;
  /** @type {boolean} */
  this[_0x34b6[1809]] = false;
  /** @type {null} */
  this[_0x34b6[1812]] = null;
  this[_0x34b6[1816]](LSELoadStatus.LS_UNLOAD);
};
/**
 * @param {?} local$$53009
 * @return {?}
 */
LSJPageLODNode[_0x34b6[219]][_0x34b6[1850]] = function(local$$53009) {
  /** @type {number} */
  var local$$53012 = 0;
  var local$$53020 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$53012 < local$$53020; local$$53012++) {
    var local$$53029 = this[_0x34b6[684]][local$$53012];
    if (local$$53029 != null) {
      if (local$$53029[_0x34b6[1841]](local$$53009) && local$$53029[_0x34b6[684]][_0x34b6[223]] > 1) {
        local$$53029[_0x34b6[1814]](true);
        var local$$53057 = local$$53029[_0x34b6[684]][0];
        if (local$$53057) {
          if (local$$53057[_0x34b6[1785]] != _0x34b6[381]) {
            if (local$$53057[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
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
LSJPageLODNode[_0x34b6[219]][_0x34b6[1261]] = function(local$$53100) {
  /** @type {boolean} */
  this[_0x34b6[1446]][_0x34b6[330]] = false;
  var local$$53120 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  var local$$53123 = 0;
  /** @type {number} */
  local$$53123 = 0;
  for (; local$$53123 < local$$53120; local$$53123++) {
    /** @type {boolean} */
    this[_0x34b6[1446]][_0x34b6[684]][local$$53123][_0x34b6[330]] = false;
  }
  /** @type {boolean} */
  this[_0x34b6[1804]] = false;
  if (!this[_0x34b6[1841]](local$$53100)) {
    /** @type {boolean} */
    this[_0x34b6[1446]][_0x34b6[330]] = false;
    return false;
  }
  this[_0x34b6[1769]](this[_0x34b6[1761]][_0x34b6[1770]]());
  this[_0x34b6[1771]](this[_0x34b6[1761]][_0x34b6[1772]]());
  if (this[_0x34b6[1785]] != _0x34b6[381]) {
    if (this[_0x34b6[1792]]() == LSELoadStatus[_0x34b6[1741]]) {
      this[_0x34b6[1822]]();
    }
    if (this[_0x34b6[1792]]() == LSELoadStatus[_0x34b6[1845]]) {
      this[_0x34b6[1060]]();
    }
    if (this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
      this[_0x34b6[1761]][_0x34b6[1758]]++;
      return false;
    }
  }
  /** @type {number} */
  var local$$53258 = 0;
  if (this[_0x34b6[1803]][_0x34b6[223]] > 0) {
    if (this[_0x34b6[1808]] == LSERangeMode[_0x34b6[1851]]) {
      if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
        local$$53258 = LSJMath[_0x34b6[1700]](this[_0x34b6[1759]][_0x34b6[658]], this[_0x34b6[1761]][_0x34b6[1765]]());
      }
    } else {
      if (this[_0x34b6[1808]] == LSERangeMode[_0x34b6[1852]]) {
        if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
          /** @type {number} */
          local$$53258 = LSJMath[_0x34b6[1705]](this[_0x34b6[1759]], this[_0x34b6[1761]][_0x34b6[1763]]()) * .5;
        }
      }
    }
  }
  /** @type {boolean} */
  var local$$53348 = true;
  /** @type {number} */
  var local$$53351 = 0;
  local$$53120 = this[_0x34b6[684]][_0x34b6[223]];
  if (this[_0x34b6[1803]][_0x34b6[223]] > 0) {
    /** @type {number} */
    local$$53123 = 0;
    for (; local$$53123 < local$$53120; local$$53123++) {
      var local$$53379 = this[_0x34b6[684]][local$$53123];
      if (local$$53123 < this[_0x34b6[1803]][_0x34b6[223]]) {
        var local$$53392 = this[_0x34b6[1803]][local$$53123];
        if (local$$53379 && local$$53258 >= local$$53392[_0x34b6[290]] && local$$53258 < local$$53392[_0x34b6[291]]) {
          if (local$$53379[_0x34b6[1261]](local$$53100)) {
            /** @type {boolean} */
            this[_0x34b6[1804]] = true;
          }
        }
      } else {
        if (local$$53379 && local$$53379[_0x34b6[1261]](local$$53100)) {
          /** @type {boolean} */
          this[_0x34b6[1804]] = true;
        }
      }
    }
    if (!this[_0x34b6[1804]] && local$$53120 > 0) {
      local$$53379 = this[_0x34b6[684]][0];
      if (local$$53379 && local$$53379[_0x34b6[1261]](local$$53100)) {
        /** @type {boolean} */
        this[_0x34b6[1804]] = true;
      }
    }
  } else {
    /** @type {number} */
    local$$53123 = 0;
    for (; local$$53123 < local$$53120; local$$53123++) {
      local$$53379 = this[_0x34b6[684]][local$$53123];
      if (local$$53379 && local$$53379[_0x34b6[1261]](local$$53100)) {
        /** @type {boolean} */
        this[_0x34b6[1804]] = true;
      }
    }
  }
  /** @type {boolean} */
  this[_0x34b6[1446]][_0x34b6[330]] = true;
  /** @type {boolean} */
  var local$$53507 = false;
  var local$$53513 = this[_0x34b6[1847]]();
  if (!this[_0x34b6[1804]] && this[_0x34b6[1761]][_0x34b6[1755]] < this[_0x34b6[1761]][_0x34b6[1754]]) {
    local$$53120 = this[_0x34b6[1811]][_0x34b6[223]];
    /** @type {number} */
    local$$53123 = 0;
    for (; local$$53123 < local$$53120; local$$53123++) {
      var local$$53551 = this[_0x34b6[1811]][local$$53123];
      if (local$$53551[_0x34b6[1045]] == LSELoadStatus[_0x34b6[1741]]) {
        this[_0x34b6[1089]](local$$53551, this[_0x34b6[1761]]);
      }
    }
  }
  local$$53120 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$53123 = 0;
  for (; local$$53123 < local$$53120; local$$53123++) {
    var local$$53599 = this[_0x34b6[1446]][_0x34b6[684]][local$$53123];
    if (local$$53599 && local$$53599 instanceof THREE[_0x34b6[329]]) {
      if (this[_0x34b6[1759]][_0x34b6[1693]] <= 0 && local$$53599[_0x34b6[1126]][_0x34b6[1447]] != null) {
        LSJMath[_0x34b6[1451]](this[_0x34b6[1761]][_0x34b6[1759]], local$$53599[_0x34b6[1126]][_0x34b6[1447]]);
      }
      if (local$$53513) {
        /** @type {boolean} */
        local$$53599[_0x34b6[330]] = true;
        /** @type {boolean} */
        local$$53599[_0x34b6[268]][_0x34b6[1853]] = displayMode === DisplayMode[_0x34b6[1854]];
        /** @type {boolean} */
        local$$53507 = true;
      } else {
        /** @type {boolean} */
        local$$53599[_0x34b6[330]] = false;
      }
    }
  }
  if (!local$$53507) {
    this[_0x34b6[1446]][_0x34b6[330]] = this[_0x34b6[1804]];
    return this[_0x34b6[1804]];
  }
  /** @type {boolean} */
  this[_0x34b6[1804]] = true;
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
  this[_0x34b6[224]] = true;
  this[_0x34b6[368]] = undefined;
  this[_0x34b6[239]] = local$$53715;
  /** @type {!Array} */
  this[_0x34b6[1855]] = [];
  /** @type {number} */
  this[_0x34b6[1856]] = 0;
  this[_0x34b6[1126]] = new THREE.SphereGeometry(5, 50, 50);
  /** @type {number} */
  this[_0x34b6[1857]] = 200;
  var local$$53769 = _0x34b6[1858];
  var local$$53771 = this;
  this[_0x34b6[1859]] = new THREE.TransformControls(local$$53714, local$$53716);
  local$$53715[_0x34b6[274]](this[_0x34b6[1859]]);
  var local$$53788;
  /**
   * @return {undefined}
   */
  local$$53771[_0x34b6[1860]] = function() {
    local$$53771[_0x34b6[1861]]();
    local$$53771[_0x34b6[1862]]();
  };
  /**
   * @return {undefined}
   */
  local$$53771[_0x34b6[1862]] = function() {
    /** @type {number} */
    local$$53788 = setTimeout(function() {
      local$$53771[_0x34b6[1859]][_0x34b6[1586]](local$$53771[_0x34b6[1859]][_0x34b6[368]]);
    }, 2500);
  };
  /**
   * @return {undefined}
   */
  local$$53771[_0x34b6[1861]] = function() {
    if (local$$53771[_0x34b6[1863]]) {
      local$$53771[_0x34b6[1864]](local$$53771[_0x34b6[1863]]);
    }
  };
  this[_0x34b6[1859]][_0x34b6[1423]](_0x34b6[1575], function(local$$53866) {
    local$$53771[_0x34b6[1861]]();
  });
  this[_0x34b6[1859]][_0x34b6[1423]](_0x34b6[1576], function(local$$53885) {
    local$$53771[_0x34b6[1861]]();
  });
  this[_0x34b6[1859]][_0x34b6[1423]](_0x34b6[1577], function(local$$53904) {
    local$$53771[_0x34b6[1860]]();
  });
  this[_0x34b6[1859]][_0x34b6[1423]](_0x34b6[1578], function(local$$53923) {
    /** @type {!Array} */
    local$$53771[_0x34b6[368]][_0x34b6[1865]] = [];
    /** @type {number} */
    i = 0;
    for (; i < local$$53771[_0x34b6[1856]]; i++) {
      local$$53771[_0x34b6[368]][_0x34b6[1865]][_0x34b6[220]](local$$53771[_0x34b6[1855]][i][_0x34b6[430]]);
    }
    local$$53771[_0x34b6[1866]]();
  });
  this[_0x34b6[1867]] = new THREE.DragControls(local$$53714, this[_0x34b6[1855]], local$$53716);
  this[_0x34b6[1867]][_0x34b6[1170]](_0x34b6[1440], function(local$$53992) {
    local$$53771[_0x34b6[1859]][_0x34b6[1585]](local$$53992[_0x34b6[368]]);
    local$$53771[_0x34b6[1861]]();
  });
  this[_0x34b6[1867]][_0x34b6[1170]](_0x34b6[1441], function(local$$54022) {
    if (local$$54022) {
      local$$53771[_0x34b6[1860]]();
    }
  });
  /**
   * @param {?} local$$54038
   * @return {undefined}
   */
  this[_0x34b6[1868]] = function(local$$54038) {
    var local$$54055 = new THREE.Mesh(this[_0x34b6[1126]], new THREE.MeshLambertMaterial({
      color : Math[_0x34b6[931]]() * 16777215
    }));
    local$$54055[_0x34b6[268]][_0x34b6[1869]] = local$$54055[_0x34b6[268]][_0x34b6[245]];
    if (local$$54038) {
      local$$54055[_0x34b6[430]][_0x34b6[338]](local$$54038);
    }
    /** @type {boolean} */
    local$$54055[_0x34b6[1714]] = true;
    /** @type {boolean} */
    local$$54055[_0x34b6[1715]] = true;
    this[_0x34b6[239]][_0x34b6[274]](local$$54055);
    this[_0x34b6[1855]][_0x34b6[220]](local$$54055);
  };
  /**
   * @param {?} local$$54117
   * @return {undefined}
   */
  this[_0x34b6[1870]] = function(local$$54117) {
    if (!this[_0x34b6[224]]) {
      return;
    }
    this[_0x34b6[1856]]++;
    local$$53771[_0x34b6[368]][_0x34b6[1865]][_0x34b6[220]](local$$54117);
    this[_0x34b6[1868]](local$$54117);
    this[_0x34b6[1866]]();
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1871]] = function() {
    if (this[_0x34b6[1856]] <= 0 || !this[_0x34b6[224]]) {
      return;
    }
    this[_0x34b6[1856]]--;
    local$$53771[_0x34b6[368]][_0x34b6[1865]][_0x34b6[1849]]();
    this[_0x34b6[239]][_0x34b6[1448]](this[_0x34b6[1855]][_0x34b6[1849]]());
    this[_0x34b6[1866]]();
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1866]] = function() {
    if (this[_0x34b6[1856]] < 2) {
      return;
    }
    var local$$54228;
    var local$$54236 = this[_0x34b6[368]][_0x34b6[1872]];
    /** @type {number} */
    var local$$54239 = 0;
    for (; local$$54239 < this[_0x34b6[1857]]; local$$54239++) {
      local$$54228 = local$$54236[_0x34b6[1126]][_0x34b6[1125]][local$$54239];
      local$$54228[_0x34b6[338]](this[_0x34b6[368]][_0x34b6[1873]](local$$54239 / (this[_0x34b6[1857]] - 1)));
    }
    /** @type {boolean} */
    local$$54236[_0x34b6[1126]][_0x34b6[1874]] = true;
  };
  /**
   * @param {?} local$$54293
   * @return {undefined}
   */
  this[_0x34b6[1587]] = function(local$$54293) {
    local$$53769 = local$$54293 ? local$$54293 : local$$53769;
    /** @type {boolean} */
    var local$$54304 = local$$53769 == _0x34b6[1858] ? true : false;
    /** @type {number} */
    i = 0;
    for (; i < local$$53771[_0x34b6[1856]]; i++) {
      /** @type {boolean} */
      local$$53771[_0x34b6[1855]][i][_0x34b6[330]] = local$$54304;
    }
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1261]] = function() {
    this[_0x34b6[1859]][_0x34b6[1261]]();
  };
  /**
   * @param {?} local$$54349
   * @return {undefined}
   */
  this[_0x34b6[1585]] = function(local$$54349) {
    this[_0x34b6[368]] = local$$54349;
    /** @type {boolean} */
    this[_0x34b6[224]] = true;
    this[_0x34b6[1856]] = local$$53771[_0x34b6[368]][_0x34b6[1865]][_0x34b6[223]];
    /** @type {number} */
    var local$$54377 = 0;
    for (; local$$54377 < this[_0x34b6[1856]]; local$$54377++) {
      this[_0x34b6[1868]](local$$53771[_0x34b6[368]][_0x34b6[1865]][local$$54377]);
    }
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1586]] = function() {
    this[_0x34b6[368]] = undefined;
    /** @type {boolean} */
    this[_0x34b6[224]] = false;
  };
};
/**
 * @param {?} local$$54429
 * @return {undefined}
 */
LSJFlyAroundCenterControls = function(local$$54429) {
  this[_0x34b6[240]] = local$$54429;
  this[_0x34b6[1875]] = new THREE.Vector3(0, 0, 0);
  /** @type {boolean} */
  this[_0x34b6[359]] = false;
  /** @type {number} */
  this[_0x34b6[1876]] = .1;
  /** @type {number} */
  this[_0x34b6[1877]] = 0;
  var local$$54464 = this;
  /**
   * @param {?} local$$54469
   * @param {?} local$$54470
   * @return {undefined}
   */
  this[_0x34b6[1878]] = function(local$$54469, local$$54470) {
    this[_0x34b6[1875]] = local$$54469;
    /** @type {boolean} */
    this[_0x34b6[359]] = true;
    this[_0x34b6[1876]] = local$$54470;
    var local$$54490 = new THREE.Vector3;
    local$$54490[_0x34b6[338]](this[_0x34b6[240]][_0x34b6[430]]);
    local$$54490[_0x34b6[1434]](this[_0x34b6[1875]]);
    /** @type {number} */
    local$$54490[_0x34b6[1287]] = 0;
    var local$$54521 = local$$54490[_0x34b6[223]]();
    if (local$$54521 == 0) {
      /** @type {number} */
      this[_0x34b6[1877]] = 0;
      return;
    }
    if (local$$54490[_0x34b6[290]] > 0) {
      if (local$$54490[_0x34b6[291]] > 0) {
        this[_0x34b6[1877]] = THREE[_0x34b6[1365]][_0x34b6[1880]](Math[_0x34b6[1879]](Math[_0x34b6[1525]](local$$54490[_0x34b6[290]]) / local$$54521));
      } else {
        /** @type {number} */
        this[_0x34b6[1877]] = 180 - THREE[_0x34b6[1365]][_0x34b6[1880]](Math[_0x34b6[1879]](Math[_0x34b6[1525]](local$$54490[_0x34b6[290]]) / local$$54521));
      }
    } else {
      if (local$$54490[_0x34b6[291]] > 0) {
        /** @type {number} */
        this[_0x34b6[1877]] = -THREE[_0x34b6[1365]][_0x34b6[1880]](Math[_0x34b6[1879]](Math[_0x34b6[1525]](local$$54490[_0x34b6[290]]) / local$$54521));
      } else {
        this[_0x34b6[1877]] = 180 + THREE[_0x34b6[1365]][_0x34b6[1880]](Math[_0x34b6[1879]](Math[_0x34b6[1525]](local$$54490[_0x34b6[290]]) / local$$54521));
      }
    }
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[661]] = function() {
    /** @type {boolean} */
    this[_0x34b6[359]] = false;
    this[_0x34b6[1875]] = undefined;
    /** @type {boolean} */
    this[_0x34b6[359]] = false;
    /** @type {number} */
    this[_0x34b6[1876]] = .1;
    /** @type {number} */
    this[_0x34b6[1877]] = 0;
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1261]] = function() {
    if (!this[_0x34b6[359]]) {
      return;
    }
    this[_0x34b6[1877]] += this[_0x34b6[1876]];
    var local$$54723 = new THREE.Vector3;
    local$$54723[_0x34b6[338]](this[_0x34b6[240]][_0x34b6[430]]);
    local$$54723[_0x34b6[1434]](this[_0x34b6[1875]]);
    /** @type {number} */
    local$$54723[_0x34b6[1287]] = 0;
    var local$$54754 = local$$54723[_0x34b6[223]]();
    /** @type {number} */
    var local$$54771 = local$$54754 * Math[_0x34b6[1562]](THREE[_0x34b6[1365]][_0x34b6[1881]](this[_0x34b6[1877]]));
    /** @type {number} */
    var local$$54788 = local$$54754 * Math[_0x34b6[349]](THREE[_0x34b6[1365]][_0x34b6[1881]](this[_0x34b6[1877]]));
    this[_0x34b6[240]][_0x34b6[430]][_0x34b6[290]] = this[_0x34b6[1875]][_0x34b6[290]] + local$$54771;
    this[_0x34b6[240]][_0x34b6[430]][_0x34b6[291]] = this[_0x34b6[1875]][_0x34b6[291]] + local$$54788;
    this[_0x34b6[240]][_0x34b6[740]][_0x34b6[1549]](this[_0x34b6[240]][_0x34b6[430]], this[_0x34b6[1875]], new THREE.Vector3(0, 0, 1));
    this[_0x34b6[240]][_0x34b6[1271]][_0x34b6[1550]](local$$54464[_0x34b6[240]][_0x34b6[740]], local$$54464[_0x34b6[240]][_0x34b6[1271]][_0x34b6[1882]]);
    this[_0x34b6[240]][_0x34b6[1263]]();
  };
};
/**
 * @param {?} local$$54896
 * @return {undefined}
 */
LSJFlyToCameraControls = function(local$$54896) {
  this[_0x34b6[240]] = local$$54896;
  this[_0x34b6[1883]] = undefined;
  /** @type {number} */
  this[_0x34b6[1884]] = 0;
  /** @type {number} */
  this[_0x34b6[1885]] = 0;
  /** @type {number} */
  this[_0x34b6[1886]] = 0;
  /** @type {number} */
  this[_0x34b6[1245]] = 5 * 1E3;
  /** @type {boolean} */
  this[_0x34b6[359]] = false;
  /** @type {number} */
  this[_0x34b6[1251]] = 0;
  var local$$54946 = this;
  this[_0x34b6[1246]] = undefined;
  this[_0x34b6[1247]] = undefined;
  this[_0x34b6[1248]] = undefined;
  this[_0x34b6[1249]] = undefined;
  /**
   * @param {?} local$$54971
   * @param {?} local$$54972
   * @param {?} local$$54973
   * @param {?} local$$54974
   * @param {?} local$$54975
   * @return {undefined}
   */
  this[_0x34b6[1887]] = function(local$$54971, local$$54972, local$$54973, local$$54974, local$$54975) {
    this[_0x34b6[1883]] = local$$54971;
    this[_0x34b6[1884]] = local$$54972;
    this[_0x34b6[1885]] = local$$54973;
    this[_0x34b6[1886]] = local$$54974;
    this[_0x34b6[1245]] = local$$54975;
    /** @type {!Array} */
    var local$$55003 = [];
    /** @type {!Array} */
    var local$$55006 = [];
    local$$55003[_0x34b6[220]](0);
    this[_0x34b6[240]][_0x34b6[430]][_0x34b6[1254]](local$$55006, local$$55006[_0x34b6[223]]);
    local$$55003[_0x34b6[220]](this[_0x34b6[1245]]);
    local$$54971[_0x34b6[1254]](local$$55006, local$$55006[_0x34b6[223]]);
    this[_0x34b6[1246]] = new THREE.VectorKeyframeTrack(_0x34b6[1255], local$$55003, local$$55006);
    this[_0x34b6[1248]] = this[_0x34b6[1246]][_0x34b6[1256]](undefined);
    /** @type {!Array} */
    var local$$55066 = [];
    this[_0x34b6[240]][_0x34b6[1239]][_0x34b6[1254]](local$$55066, local$$55066[_0x34b6[223]]);
    var local$$55095 = new THREE.Euler(this[_0x34b6[1884]], this[_0x34b6[1885]], this[_0x34b6[1886]], _0x34b6[1560]);
    var local$$55099 = new THREE.Quaternion;
    local$$55099[_0x34b6[1554]](local$$55095, true);
    local$$55099[_0x34b6[1254]](local$$55066, local$$55066[_0x34b6[223]]);
    this[_0x34b6[1247]] = new THREE.QuaternionKeyframeTrack(_0x34b6[1257], local$$55003, local$$55066);
    this[_0x34b6[1249]] = this[_0x34b6[1247]][_0x34b6[1256]](undefined);
    /** @type {boolean} */
    this[_0x34b6[359]] = true;
    this[_0x34b6[1251]] = Date[_0x34b6[348]]();
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[661]] = function() {
    this[_0x34b6[1883]] = undefined;
    /** @type {number} */
    this[_0x34b6[1884]] = 0;
    /** @type {number} */
    this[_0x34b6[1885]] = 0;
    /** @type {number} */
    this[_0x34b6[1886]] = 0;
    /** @type {number} */
    this[_0x34b6[1245]] = 5 * 1E3;
    /** @type {boolean} */
    this[_0x34b6[359]] = false;
    /** @type {number} */
    this[_0x34b6[1251]] = 0;
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1261]] = function() {
    if (!this[_0x34b6[359]]) {
      return;
    }
    var local$$55223 = Date[_0x34b6[348]]();
    if (local$$55223 - this[_0x34b6[1251]] > this[_0x34b6[1245]]) {
      /** @type {boolean} */
      this[_0x34b6[359]] = false;
      local$$54946[_0x34b6[240]][_0x34b6[430]][_0x34b6[290]] = this[_0x34b6[1883]][_0x34b6[290]];
      local$$54946[_0x34b6[240]][_0x34b6[430]][_0x34b6[291]] = this[_0x34b6[1883]][_0x34b6[291]];
      local$$54946[_0x34b6[240]][_0x34b6[430]][_0x34b6[1287]] = this[_0x34b6[1883]][_0x34b6[1287]];
      return;
    }
    var local$$55305 = this[_0x34b6[1248]][_0x34b6[1262]](local$$55223 - this[_0x34b6[1251]]);
    local$$54946[_0x34b6[240]][_0x34b6[430]][_0x34b6[462]](local$$55305);
    var local$$55329 = this[_0x34b6[1249]][_0x34b6[1262]](local$$55223 - this[_0x34b6[1251]]);
    local$$54946[_0x34b6[240]][_0x34b6[1239]][_0x34b6[462]](local$$55329);
    local$$54946[_0x34b6[240]][_0x34b6[1263]]();
  };
};
/**
 * @param {?} local$$55361
 * @return {undefined}
 */
LSJFlyWithLineControls = function(local$$55361) {
  this[_0x34b6[368]] = undefined;
  this[_0x34b6[1888]] = undefined;
  /** @type {boolean} */
  this[_0x34b6[359]] = false;
  this[_0x34b6[1889]] = local$$55361;
  this[_0x34b6[1890]] = new THREE.Vector3;
  this[_0x34b6[570]] = new THREE.Vector3;
  /** @type {boolean} */
  this[_0x34b6[1891]] = true;
  /** @type {number} */
  this[_0x34b6[1245]] = 20 * 1E3;
  this[_0x34b6[1251]] = Date[_0x34b6[348]]();
  var local$$55421 = this;
  /**
   * @return {?}
   */
  this[_0x34b6[1892]] = function() {
    var local$$55428 = {};
    if (this[_0x34b6[368]] == undefined) {
      return local$$55428;
    }
    /** @type {!Array} */
    var local$$55439 = [];
    /** @type {number} */
    var local$$55442 = 0;
    for (; local$$55442 < this[_0x34b6[368]][_0x34b6[1865]][_0x34b6[223]]; local$$55442++) {
      var local$$55463 = this[_0x34b6[368]][_0x34b6[1865]][local$$55442];
      local$$55439[_0x34b6[220]](local$$55463[_0x34b6[290]], local$$55463[_0x34b6[291]], local$$55463[_0x34b6[1287]]);
    }
    var local$$55483 = {};
    local$$55483[_0x34b6[1893]] = this[_0x34b6[1245]];
    local$$55483[_0x34b6[1894]] = {
      coordinates : local$$55439
    };
    local$$55428[_0x34b6[1895]] = {
      name : _0x34b6[1896],
      playlist : local$$55483
    };
    return local$$55428;
  };
  /**
   * @param {string} local$$55517
   * @param {string} local$$55518
   * @return {undefined}
   */
  this[_0x34b6[1585]] = function(local$$55517, local$$55518) {
    /** @type {string} */
    this[_0x34b6[368]] = local$$55517;
    if (local$$55517 != undefined && local$$55421[_0x34b6[368]][_0x34b6[1865]][_0x34b6[223]] > 1) {
      this[_0x34b6[1888]] = new THREE.TubeGeometry(local$$55517, 200, 2, 1, false);
    }
    /** @type {boolean} */
    this[_0x34b6[359]] = true;
    if (local$$55518 != undefined) {
      /** @type {string} */
      this[_0x34b6[1245]] = local$$55518;
    }
    this[_0x34b6[1251]] = Date[_0x34b6[348]]();
    this[_0x34b6[1261]]();
  };
  /**
   * @return {undefined}
   */
  this[_0x34b6[1586]] = function() {
    this[_0x34b6[368]] = undefined;
    this[_0x34b6[1888]] = undefined;
    /** @type {boolean} */
    this[_0x34b6[359]] = false;
  };
  /**
   * @param {number} local$$55613
   * @return {undefined}
   */
  this[_0x34b6[1261]] = function(local$$55613) {
    if (!this[_0x34b6[359]]) {
      return;
    }
    if (local$$55421[_0x34b6[368]] === undefined) {
      return;
    }
    if (this[_0x34b6[1888]] == undefined && local$$55421[_0x34b6[368]][_0x34b6[1865]][_0x34b6[223]] > 1) {
      this[_0x34b6[1888]] = new THREE.TubeGeometry(this[_0x34b6[368]], 200, 2, 1, false);
    }
    if (this[_0x34b6[1888]] === undefined) {
      return;
    }
    var local$$55676 = Date[_0x34b6[348]]();
    if (local$$55613 < 0 || local$$55613 == 0 || local$$55613 == undefined) {
      /** @type {number} */
      local$$55613 = 1;
    }
    /** @type {number} */
    var local$$55695 = this[_0x34b6[1245]] / local$$55613;
    /** @type {number} */
    var local$$55703 = (local$$55676 - this[_0x34b6[1251]]) % local$$55695 / local$$55695;
    var local$$55718 = local$$55421[_0x34b6[1888]][_0x34b6[1898]][_0x34b6[1058]][_0x34b6[1897]](local$$55703);
    var local$$55729 = local$$55421[_0x34b6[1888]][_0x34b6[1899]][_0x34b6[223]];
    /** @type {number} */
    var local$$55732 = local$$55703 * local$$55729;
    var local$$55738 = Math[_0x34b6[376]](local$$55732);
    /** @type {number} */
    var local$$55743 = (local$$55738 + 1) % local$$55729;
    local$$55421[_0x34b6[1890]][_0x34b6[1697]](local$$55421[_0x34b6[1888]][_0x34b6[1900]][local$$55743], local$$55421[_0x34b6[1888]][_0x34b6[1900]][local$$55738]);
    local$$55421[_0x34b6[1890]][_0x34b6[350]](local$$55732 - local$$55738)[_0x34b6[274]](local$$55421[_0x34b6[1888]][_0x34b6[1900]][local$$55738]);
    var local$$55800 = local$$55421[_0x34b6[1888]][_0x34b6[1898]][_0x34b6[1058]][_0x34b6[1901]](local$$55703);
    local$$55421[_0x34b6[570]][_0x34b6[338]](local$$55421[_0x34b6[1890]])[_0x34b6[1599]](local$$55800);
    local$$55421[_0x34b6[1889]][_0x34b6[430]][_0x34b6[338]](local$$55718);
    /** @type {number} */
    var local$$55844 = local$$55703 + 30 / local$$55421[_0x34b6[1888]][_0x34b6[1898]][_0x34b6[1058]][_0x34b6[1902]]();
    /** @type {number} */
    local$$55844 = local$$55844 > 1 ? 1 : local$$55844;
    var local$$55865 = local$$55421[_0x34b6[1888]][_0x34b6[1898]][_0x34b6[1058]][_0x34b6[1897]](local$$55844);
    local$$55865[_0x34b6[1287]] -= 20;
    local$$55421[_0x34b6[1889]][_0x34b6[740]][_0x34b6[1549]](local$$55421[_0x34b6[1889]][_0x34b6[430]], local$$55865, new THREE.Vector3(0, 0, 1));
    local$$55421[_0x34b6[1889]][_0x34b6[1271]][_0x34b6[1550]](local$$55421[_0x34b6[1889]][_0x34b6[740]], local$$55421[_0x34b6[1889]][_0x34b6[1271]][_0x34b6[1882]]);
  };
};
/**
 * @return {undefined}
 */
LSJClippingControl = function() {
  /** @type {boolean} */
  this[_0x34b6[1903]] = false;
  /** @type {boolean} */
  this[_0x34b6[1904]] = false;
  /** @type {boolean} */
  this[_0x34b6[1905]] = false;
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {!Array} */
  this[_0x34b6[1906]] = [];
  /** @type {!Array} */
  this[_0x34b6[1907]] = [];
  /** @type {number} */
  this[_0x34b6[1508]] = 0;
  this[_0x34b6[1908]] = new THREE.SphereGeometry(.4, 10, 10);
  this[_0x34b6[245]] = new THREE.Color(16711680);
};
/** @type {function(): undefined} */
LSJClippingControl[_0x34b6[219]][_0x34b6[1183]] = LSJClippingControl;
/**
 * @param {?} local$$56012
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1909]] = function(local$$56012) {
  this[_0x34b6[1906]][_0x34b6[220]](local$$56012[_0x34b6[212]]());
};
/**
 * @param {?} local$$56036
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1910]] = function(local$$56036) {
  this[_0x34b6[1906]][this[_0x34b6[1906]][_0x34b6[223]] - 1] = local$$56036;
};
/**
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[235]] = function() {
  /** @type {number} */
  this[_0x34b6[1508]] = 0;
  /** @type {!Array} */
  this[_0x34b6[1906]] = [];
  /** @type {!Array} */
  this[_0x34b6[1907]] = [];
  getScene()[_0x34b6[1448]](this[_0x34b6[1446]]);
  /** @type {boolean} */
  this[_0x34b6[1904]] = false;
  /** @type {boolean} */
  this[_0x34b6[1905]] = false;
  /** @type {!Array} */
  controlRender[_0x34b6[1911]] = [];
};
/**
 * @param {?} local$$56119
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1261]] = function(local$$56119) {
  local$$56119[_0x34b6[1448]](this[_0x34b6[1446]]);
  /** @type {!Array} */
  this[_0x34b6[1907]] = [];
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {number} */
  var local$$56143 = 0;
  for (; local$$56143 < this[_0x34b6[1906]][_0x34b6[223]]; local$$56143++) {
    var local$$56161 = new THREE.Mesh(this[_0x34b6[1908]], createSphereMaterial());
    var local$$56175 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$56161[_0x34b6[1912]]());
    var local$$56197 = projectedRadius(1, getCamera()[_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$56175, getRenderer()[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    var local$$56201 = 5 / local$$56197;
    local$$56161[_0x34b6[430]][_0x34b6[338]](this[_0x34b6[1906]][local$$56143]);
    local$$56161[_0x34b6[1090]][_0x34b6[334]](local$$56201, local$$56201, local$$56201);
    this[_0x34b6[1907]][_0x34b6[220]](local$$56161);
    this[_0x34b6[1446]][_0x34b6[274]](local$$56161);
  }
  var local$$56244 = new THREE.Geometry;
  /** @type {number} */
  local$$56143 = 0;
  for (; local$$56143 < this[_0x34b6[1906]][_0x34b6[223]] - 1; local$$56143++) {
    local$$56244[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][local$$56143]);
    local$$56244[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][local$$56143 + 1]);
  }
  local$$56244[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][0]);
  local$$56244[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][this[_0x34b6[1906]][_0x34b6[223]] - 1]);
  var local$$56327 = new THREE.LineBasicMaterial({
    color : 65280,
    linewidth : 2
  });
  var local$$56331 = new THREE.Line(local$$56244, local$$56327);
  this[_0x34b6[1446]][_0x34b6[274]](local$$56331);
  local$$56119[_0x34b6[274]](this[_0x34b6[1446]]);
};
/**
 * @param {?} local$$56359
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1914]] = function(local$$56359) {
  local$$56359[_0x34b6[1448]](this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[1849]]());
  var local$$56378 = new THREE.Geometry;
  /** @type {number} */
  var local$$56381 = 0;
  for (; local$$56381 < this[_0x34b6[1906]][_0x34b6[223]] - 1; local$$56381++) {
    local$$56378[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][local$$56381]);
    local$$56378[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][local$$56381 + 1]);
  }
  local$$56378[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][0]);
  local$$56378[_0x34b6[1125]][_0x34b6[220]](this[_0x34b6[1906]][this[_0x34b6[1906]][_0x34b6[223]] - 1]);
  var local$$56461 = new THREE.LineBasicMaterial({
    color : 65280,
    linewidth : 2
  });
  var local$$56465 = new THREE.Line(local$$56378, local$$56461);
  this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[220]](local$$56465);
  local$$56359[_0x34b6[274]](this[_0x34b6[1446]]);
};
/**
 * @param {?} local$$56496
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1915]] = function(local$$56496) {
  if (intersectsObj[_0x34b6[223]] != 0 && this[_0x34b6[1905]] == false) {
    if (local$$56496[_0x34b6[1916]] == LSJMOUSEBUTTON[_0x34b6[655]] || local$$56496[_0x34b6[1916]] == LSJMOUSEBUTTON[_0x34b6[573]]) {
      /** @type {boolean} */
      dragControls[_0x34b6[224]] = false;
    } else {
      /** @type {boolean} */
      dragControls[_0x34b6[224]] = true;
      /** @type {!Array} */
      this[_0x34b6[1906]] = [];
      /** @type {number} */
      var local$$56547 = 0;
      for (; local$$56547 < this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]] - 1; local$$56547++) {
        this[_0x34b6[1906]][_0x34b6[220]](this[_0x34b6[1446]][_0x34b6[684]][local$$56547][_0x34b6[430]]);
      }
      this[_0x34b6[1914]](getScene());
    }
  }
  if (this[_0x34b6[1905]] == true) {
    var local$$56608 = intersectSceneAndPlane(local$$56496[_0x34b6[1429]], local$$56496[_0x34b6[1430]]);
    if (!LSJMath[_0x34b6[1704]](local$$56608)) {
      this[_0x34b6[1910]](local$$56608);
      this[_0x34b6[1261]](getScene());
    }
  }
};
/**
 * @param {?} local$$56640
 * @param {?} local$$56641
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1917]] = function(local$$56640, local$$56641) {
  if (this[_0x34b6[1904]]) {
    this[_0x34b6[1508]] = this[_0x34b6[1508]] + 1;
    if (this[_0x34b6[1508]] == 1) {
      this[_0x34b6[1909]](local$$56641);
      this[_0x34b6[1909]](local$$56641);
      /** @type {boolean} */
      this[_0x34b6[1905]] = true;
    }
    if (this[_0x34b6[1508]] > 1) {
      this[_0x34b6[1910]](local$$56641);
      this[_0x34b6[1909]](local$$56641);
    }
  }
};
/**
 * @param {?} local$$56709
 * @param {?} local$$56710
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1918]] = function(local$$56709, local$$56710) {
  dragControls = new THREE.DragControls(controlCamera, this[_0x34b6[1446]][_0x34b6[684]], controlRender[_0x34b6[1694]]);
  /** @type {boolean} */
  this[_0x34b6[1905]] = false;
  /** @type {boolean} */
  this[_0x34b6[1904]] = false;
};
/**
 * @return {undefined}
 */
LSJClippingControl[_0x34b6[219]][_0x34b6[1919]] = function() {
  if (this[_0x34b6[1906]][_0x34b6[223]] > 2) {
    controlRender[_0x34b6[1911]] = this[_0x34b6[1906]];
    controlRender[_0x34b6[1903]] = this[_0x34b6[1903]];
  }
  getScene()[_0x34b6[1448]](this[_0x34b6[1446]]);
  /** @type {!Array} */
  this[_0x34b6[1906]] = [];
  /** @type {number} */
  this[_0x34b6[1508]] = 0;
  this[_0x34b6[1446]] = new THREE.Group;
};
/**
 * @return {undefined}
 */
LSJTrackLight = function() {
  /** @type {null} */
  this[_0x34b6[1920]] = null;
  this[_0x34b6[430]] = new THREE.Vector3;
  /** @type {boolean} */
  this[_0x34b6[1921]] = true;
};
LSJTrackLight[_0x34b6[219]] = Object[_0x34b6[242]](LSJTrackLight[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJTrackLight[_0x34b6[219]][_0x34b6[1183]] = LSJTrackLight;
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
  local$$57023[_0x34b6[443]] = function() {
    local$$57013[_0x34b6[554]] = this;
    /** @type {boolean} */
    local$$57013[_0x34b6[1275]] = true;
  };
  local$$57023[_0x34b6[551]] = local$$57009;
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
  controlMaxTilt = local$$57125 * Math[_0x34b6[979]] / 180;
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
  controlMinTilt = local$$57145 * Math[_0x34b6[979]] / 180;
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
  local$$57215[_0x34b6[430]][_0x34b6[334]](local$$57201, local$$57202, local$$57203);
  local$$57215[_0x34b6[1922]] = local$$57207;
  controlScene[_0x34b6[274]](local$$57215);
  local$$57215[_0x34b6[1714]] = local$$57206;
  var local$$57240 = local$$57208;
  /** @type {number} */
  local$$57215[_0x34b6[1923]] = -local$$57240;
  local$$57215[_0x34b6[1924]] = local$$57240;
  local$$57215[_0x34b6[1925]] = local$$57240;
  /** @type {number} */
  local$$57215[_0x34b6[1926]] = -local$$57240;
  local$$57215[_0x34b6[1927]] = local$$57209;
  local$$57215[_0x34b6[1928]] = local$$57210;
  /** @type {number} */
  local$$57215[_0x34b6[1929]] = 1024;
  /** @type {number} */
  local$$57215[_0x34b6[1930]] = 1024;
  /** @type {number} */
  local$$57215[_0x34b6[1931]] = -.01;
  if (local$$57206) {
    /** @type {boolean} */
    controlRender[_0x34b6[1932]][_0x34b6[224]] = true;
    if (sceneCullFace) {
      controlRender[_0x34b6[1932]][_0x34b6[1933]] = THREE[_0x34b6[1934]];
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
  if (local$$57319 == _0x34b6[1436] && orbitControls != undefined) {
    orbitControls[_0x34b6[232]]();
    orbitControls == undefined;
  } else {
    if (local$$57319 == _0x34b6[1935]) {
      controlCamera[_0x34b6[1936]][_0x34b6[334]](0, 0, 1);
      orbitControls = new THREE.OrbitControls(controlCamera, controlRender[_0x34b6[1694]]);
      /** @type {boolean} */
      orbitControls[_0x34b6[1937]] = false;
      /** @type {number} */
      orbitControls[_0x34b6[1938]] = .35;
      /** @type {boolean} */
      orbitControls[_0x34b6[1939]] = true;
      setControlMaxTilt(180);
    }
  }
}
/**
 * @return {undefined}
 */
function createModelDefaultMat() {
  lmModelDefaultMat = new THREE.MeshPhongMaterial;
  lmModelDefaultMat[_0x34b6[245]] = (new THREE.Color)[_0x34b6[1533]](.6, .6, .6);
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
  controlCamera = new THREE.PerspectiveCamera(45, window[_0x34b6[505]] / window[_0x34b6[506]], .1, 15E3);
  controlScreenCam = new THREE.OrthographicCamera(0, window[_0x34b6[505]], window[_0x34b6[506]], 0, 0, 30);
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
  stats[_0x34b6[1940]] = false;
  stats[_0x34b6[1694]][_0x34b6[428]][_0x34b6[430]] = _0x34b6[451];
  stats[_0x34b6[1694]][_0x34b6[428]][_0x34b6[434]] = _0x34b6[435];
  controlDiv[_0x34b6[412]](stats[_0x34b6[1694]]);
  controlSkyBox = new LSJSkyBox;
  controlSkyBox[_0x34b6[1610]](window[_0x34b6[505]], window[_0x34b6[506]]);
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
  movePlaneMeshs[_0x34b6[220]](movePlaneMesh);
  controlScene[_0x34b6[274]](controlFeatueLOD[_0x34b6[1446]]);
  controlScene[_0x34b6[274]](controlLayers[_0x34b6[1446]]);
  controlRender = new THREE.WebGLRenderer({
    antialias : true,
    preserveDrawingBuffer : true
  });
  controlRender[_0x34b6[262]](0);
  controlRender[_0x34b6[1942]](window[_0x34b6[1941]]);
  controlRender[_0x34b6[221]](window[_0x34b6[505]], window[_0x34b6[506]]);
  /** @type {boolean} */
  controlRender[_0x34b6[1943]] = true;
  /** @type {boolean} */
  controlRender[_0x34b6[1944]] = true;
  billboardPlugin = new LSJBillboardPlugin(controlRender, billboards);
  composer = new THREE.EffectComposer(controlRender);
  renderPass = new THREE.RenderPass(controlScene, controlCamera);
  composer[_0x34b6[1945]](renderPass);
  outlinePass = new THREE.OutlinePass(new THREE.Vector2(window[_0x34b6[505]], window[_0x34b6[506]]), controlScene, controlCamera);
  outlinePass[_0x34b6[281]] = new THREE.Color(.14, .92, .92);
  composer[_0x34b6[1945]](outlinePass);
  /**
   * @param {?} local$$57709
   * @return {undefined}
   */
  var local$$57734 = function(local$$57709) {
    outlinePass[_0x34b6[358]] = local$$57709;
    local$$57709[_0x34b6[1092]] = THREE[_0x34b6[1083]];
    local$$57709[_0x34b6[1093]] = THREE[_0x34b6[1083]];
  };
  var local$$57738 = new THREE.TextureLoader;
  local$$57738[_0x34b6[1060]](_0x34b6[1946], local$$57734);
  effectFXAA = new THREE.ShaderPass(THREE.FXAAShader);
  effectFXAA[_0x34b6[267]][_0x34b6[289]][_0x34b6[275]][_0x34b6[334]](1 / window[_0x34b6[505]], 1 / window[_0x34b6[506]]);
  /** @type {boolean} */
  effectFXAA[_0x34b6[236]] = true;
  composer[_0x34b6[1945]](effectFXAA);
  cameraMode = _0x34b6[1436];
  var local$$57793 = new THREE.Geometry;
  local$$57793[_0x34b6[1125]][_0x34b6[220]](new THREE.Vector3);
  var local$$57815 = new THREE.PointsMaterial({
    size : 40,
    sizeAttenuation : false,
    alphaTest : 0,
    depthTest : false,
    transparent : true
  });
  controlNaviIconPath = _0x34b6[1947];
  controlNaviIconMesh = new THREE.Points(local$$57793, local$$57815);
  /** @type {boolean} */
  controlNaviIconMesh[_0x34b6[330]] = false;
  controlScene[_0x34b6[274]](controlNaviIconMesh);
  var local$$57839 = new THREE.TextureLoader;
  local$$57839[_0x34b6[1060]](controlNaviIconPath, function(local$$57844) {
    local$$57815[_0x34b6[645]] = local$$57844;
  });
  controlFeatueLOD[_0x34b6[1767]]()[_0x34b6[334]](0, 0, window[_0x34b6[505]], window[_0x34b6[506]]);
  controlDiv[_0x34b6[412]](controlRender[_0x34b6[1694]]);
  editorLineControls = new EditorLineControls(controlCamera, controlScene, controlRender[_0x34b6[1694]]);
  window[_0x34b6[1423]](_0x34b6[1948], function local$$57891() {
    /** @type {number} */
    controlCamera[_0x34b6[1619]] = window[_0x34b6[505]] / window[_0x34b6[506]];
    controlScreenCam[_0x34b6[655]] = window[_0x34b6[505]];
    controlScreenCam[_0x34b6[434]] = window[_0x34b6[506]];
    controlCamera[_0x34b6[1620]]();
    controlFeatueLOD[_0x34b6[1767]]()[_0x34b6[334]](0, 0, window[_0x34b6[505]], window[_0x34b6[506]]);
    controlRender[_0x34b6[221]](window[_0x34b6[505]], window[_0x34b6[506]]);
    controlSkyBox[_0x34b6[1618]](window[_0x34b6[505]], window[_0x34b6[506]]);
    var local$$57970 = window[_0x34b6[505]] || 1;
    var local$$57977 = window[_0x34b6[506]] || 1;
    composer[_0x34b6[221]](local$$57970, local$$57977);
    effectFXAA[_0x34b6[267]][_0x34b6[289]][_0x34b6[275]][_0x34b6[334]](1 / window[_0x34b6[505]], 1 / window[_0x34b6[506]]);
  }, false);
  /** @type {number} */
  controlMaxTilt = Math[_0x34b6[979]] / 2;
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
  controlFeatueLOD[_0x34b6[1261]](controlCamera);
  if (!bFirstCamPosSet && (!controlFeatueLOD[_0x34b6[1759]][_0x34b6[1456]]() || !controlLayers[_0x34b6[1450]]()[_0x34b6[1456]]())) {
    var local$$58147 = new THREE.Vector3(0, 0, 0);
    /** @type {number} */
    var local$$58150 = 0;
    if (!controlFeatueLOD[_0x34b6[1759]][_0x34b6[1456]]()) {
      this[_0x34b6[1447]] = controlFeatueLOD[_0x34b6[1759]][_0x34b6[212]]();
      local$$58147 = controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]];
      local$$58150 = controlFeatueLOD[_0x34b6[1759]][_0x34b6[1693]];
      controlCamera[_0x34b6[430]][_0x34b6[290]] = local$$58147[_0x34b6[290]];
      controlCamera[_0x34b6[430]][_0x34b6[291]] = local$$58147[_0x34b6[291]];
      controlCamera[_0x34b6[430]][_0x34b6[1287]] = local$$58147[_0x34b6[1287]] + local$$58150;
    } else {
      this[_0x34b6[1447]] = controlLayers[_0x34b6[1450]]()[_0x34b6[212]]();
      local$$58147 = controlLayers[_0x34b6[1450]]()[_0x34b6[658]];
      local$$58150 = controlLayers[_0x34b6[1450]]()[_0x34b6[1693]];
      controlCamera[_0x34b6[430]][_0x34b6[290]] = local$$58147[_0x34b6[290]];
      controlCamera[_0x34b6[430]][_0x34b6[291]] = local$$58147[_0x34b6[291]];
      controlCamera[_0x34b6[430]][_0x34b6[1287]] = local$$58147[_0x34b6[1287]] + local$$58150 * 2;
      if (orbitControls != undefined) {
        orbitControls[_0x34b6[1875]] = local$$58147;
      }
    }
    controlCamera[_0x34b6[1549]](new THREE.Vector3(local$$58147[_0x34b6[290]], local$$58147[_0x34b6[291]], local$$58147[_0x34b6[1287]]));
    /** @type {boolean} */
    bFirstCamPosSet = true;
    try {
      if (onPageLODLoaded && typeof onPageLODLoaded == _0x34b6[391]) {
        onPageLODLoaded(local$$58147, local$$58150);
      }
    } catch (local$$58331) {
    }
  }
  local$$58147 = controlLayers[_0x34b6[1450]]()[_0x34b6[658]];
  local$$58150 = controlLayers[_0x34b6[1450]]()[_0x34b6[1693]];
  if (local$$58150 != 0 && local$$58150 > 100) {
    var local$$58364 = new THREE.Vector3;
    local$$58364[_0x34b6[1697]](controlCamera[_0x34b6[430]], local$$58147);
    if (local$$58364[_0x34b6[223]]() - local$$58150 > 0) {
      /** @type {number} */
      controlCamera[_0x34b6[345]] = 1;
    } else {
      /** @type {number} */
      controlCamera[_0x34b6[345]] = .1;
    }
    controlCamera[_0x34b6[1620]]();
  }
  rulerDistance[_0x34b6[225]](this);
  controlLayers[_0x34b6[225]](this);
  if (outlinePass[_0x34b6[280]][_0x34b6[223]] > 0) {
    /** @type {boolean} */
    controlRender[_0x34b6[258]] = true;
    controlRender[_0x34b6[262]](16773360);
    controlRender[_0x34b6[1949]](0);
    composer[_0x34b6[225]]();
  } else {
    controlRender[_0x34b6[225]](controlScene, controlCamera);
  }
  billboardPlugin[_0x34b6[225]](controlScene, controlCamera, this);
  stats[_0x34b6[1261]]();
  try {
    if (onProgressInfo && typeof onProgressInfo == _0x34b6[391]) {
      if (controlFeatueLOD[_0x34b6[1737]] > 0 && controlFeatueLOD[_0x34b6[1758]] < controlFeatueLOD[_0x34b6[1737]]) {
        /** @type {number} */
        var local$$58490 = controlFeatueLOD[_0x34b6[1758]] / controlFeatueLOD[_0x34b6[1737]];
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
  controlFeatueLOD[_0x34b6[1760]](local$$58507);
}
/**
 * @param {?} local$$58517
 * @return {undefined}
 */
function addFeaturePageLODNode1(local$$58517) {
  var local$$58520 = new LSJPageLODNode;
  local$$58520[_0x34b6[1785]] = local$$58517;
  controlFeatueLOD[_0x34b6[1760]](local$$58520);
}
/**
 * @param {?} local$$58535
 * @param {!Object} local$$58536
 * @return {undefined}
 */
function openLFP(local$$58535, local$$58536) {
  if (local$$58536 == undefined) {
    controlFeatueLOD[_0x34b6[452]](local$$58535);
  } else {
    controlFeatueLOD[_0x34b6[1787]](local$$58535);
  }
}
/**
 * @return {undefined}
 */
function animateSceneControl() {
  requestAnimationFrame(animateSceneControl);
  if (cameraMode == _0x34b6[1935]) {
    if (flyAroundCenterControls[_0x34b6[359]]) {
      orbitControls[_0x34b6[1875]] = flyAroundCenterControls[_0x34b6[1875]];
    }
    orbitControls[_0x34b6[1261]]();
  }
  flyWithLineControls[_0x34b6[1261]]();
  flyToCameraControls[_0x34b6[1261]]();
  flyAroundCenterControls[_0x34b6[1261]]();
  editorLineControls[_0x34b6[1261]]();
  cameraTrackControls[_0x34b6[1261]]();
  if (cameraMode != _0x34b6[1935]) {
    momentumScene();
  }
  inertiaScene();
  var local$$58625 = Date[_0x34b6[348]]();
  if (controlNaviLiveTime > 0) {
    if (local$$58625 - controlNaviLiveTime > 300) {
      /** @type {boolean} */
      controlNaviIconMesh[_0x34b6[330]] = false;
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
  lastControMouseScreen1[_0x34b6[290]] = local$$58651;
  /** @type {number} */
  lastControMouseScreen1[_0x34b6[291]] = local$$58652;
  /** @type {number} */
  lastControMouseScreen2[_0x34b6[290]] = local$$58653;
  /** @type {number} */
  lastControMouseScreen2[_0x34b6[291]] = local$$58654;
}
/**
 * @param {!Object} local$$58703
 * @return {undefined}
 */
function updateNaviIconMesh(local$$58703) {
  var local$$58708 = controlNaviIconMesh[_0x34b6[1126]];
  controlNaviIconMesh[_0x34b6[430]][_0x34b6[338]](local$$58703);
  controlNaviIconMesh[_0x34b6[1263]]();
  /** @type {boolean} */
  controlNaviIconMesh[_0x34b6[330]] = true;
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
  flyToCameraControls[_0x34b6[661]]();
  flyWithLineControls[_0x34b6[1586]]();
  flyAroundCenterControls[_0x34b6[661]]();
  stopSceneInertia();
  var local$$58757 = intersectSceneAndPlane(local$$58735, local$$58736);
  controlClickStart[_0x34b6[334]](local$$58735, local$$58736);
  if (!LSJMath[_0x34b6[1704]](local$$58757)) {
    mouseDWorldPos1[_0x34b6[334]](local$$58757[_0x34b6[290]], local$$58757[_0x34b6[291]], local$$58757[_0x34b6[1287]]);
    /** @type {boolean} */
    bControlStartPan = true;
    if (local$$58737 == LSJMOUSEBUTTON[_0x34b6[432]]) {
      if (updateControlDoubleClick(local$$58735, local$$58736)) {
        controlInertiaParam[_0x34b6[1950]][_0x34b6[338]](local$$58757);
        /** @type {boolean} */
        controlInertiaParam[_0x34b6[1951]] = true;
        /** @type {number} */
        controlInertiaParam[_0x34b6[1952]] = 0;
        controlInertiaParam[_0x34b6[1953]] = dControlDoubleClickZoomRatio;
        if (onControlPageLODDoubleClick != undefined) {
          var local$$58821 = new THREE.Vector2;
          /** @type {number} */
          local$$58821[_0x34b6[290]] = local$$58735 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
          /** @type {number} */
          local$$58821[_0x34b6[291]] = -(local$$58736 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
          var local$$58858 = new THREE.Raycaster;
          local$$58858[_0x34b6[1431]](local$$58821, controlCamera);
          var local$$58876 = local$$58858[_0x34b6[1437]](controlFeatueLOD[_0x34b6[1446]][_0x34b6[684]], true);
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
          if (local$$58876[_0x34b6[223]] > 0) {
            local$$58885 = local$$58876[0][_0x34b6[1435]][_0x34b6[290]];
            local$$58888 = local$$58876[0][_0x34b6[1435]][_0x34b6[291]];
            local$$58891 = local$$58876[0][_0x34b6[1435]][_0x34b6[1287]];
            local$$58882 = local$$58876[0][_0x34b6[368]];
            onControlPageLODDoubleClick(new THREE.Vector2(local$$58735, local$$58736), new THREE.Vector3(local$$58885, local$$58888, local$$58891));
          }
        }
      }
    } else {
      if (local$$58737 == LSJMOUSEBUTTON[_0x34b6[655]] || local$$58737 == LSJMOUSEBUTTON[_0x34b6[573]]) {
        updateNaviIconMesh(mouseDWorldPos1);
      }
    }
  } else {
    mouseDWorldPos1[_0x34b6[334]](0, 0, 0);
  }
  lastControMouseScreen1[_0x34b6[290]] = local$$58735;
  /** @type {!Array} */
  lastControMouseScreen1[_0x34b6[291]] = local$$58736;
  if (local$$58737 == LSJMOUSEBUTTON[_0x34b6[432]]) {
    m_lastLBDMouse[_0x34b6[334]](local$$58735, local$$58736);
  }
}
/**
 * @param {?} local$$58999
 * @return {undefined}
 */
function onLSJDivMouseDown(local$$58999) {
  local$$58999[_0x34b6[1428]]();
  if (controlNeedStopInertia) {
    stopSceneInertia();
  }
  mouseDown(local$$58999[_0x34b6[1429]], local$$58999[_0x34b6[1430]], local$$58999[_0x34b6[1916]]);
  try {
    if (onCustomMouseDown && typeof onCustomMouseDown == _0x34b6[391]) {
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
    m_firClickTime = Date[_0x34b6[348]]();
    /** @type {number} */
    m_timeGap = 0;
  } else {
    if (m_secClickTime == 0) {
      m_secClickTime = Date[_0x34b6[348]]();
      /** @type {number} */
      m_timeGap = m_secClickTime - m_firClickTime;
    }
  }
  var local$$59079 = getDist(local$$59040, local$$59041, m_lastLBDMouse[_0x34b6[290]], m_lastLBDMouse[_0x34b6[291]]);
  if (m_timeGap > 0 && m_timeGap < 500) {
    /** @type {number} */
    m_firClickTime = 0;
    /** @type {number} */
    m_secClickTime = 0;
    if (m_lastLBDMouse[_0x34b6[290]] >= 0 && local$$59040 >= 0 && local$$59079 < 20) {
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
  mouseDWorldPos1[_0x34b6[290]] = 0;
  /** @type {number} */
  mouseDWorldPos1[_0x34b6[291]] = 0;
  /** @type {number} */
  mouseDWorldPos1[_0x34b6[1287]] = 0;
  /** @type {number} */
  mouseDWorldPos2[_0x34b6[290]] = 0;
  /** @type {number} */
  mouseDWorldPos2[_0x34b6[291]] = 0;
  /** @type {number} */
  mouseDWorldPos2[_0x34b6[1287]] = 0;
  /** @type {number} */
  lastControMouseScreen1[_0x34b6[290]] = -1;
  /** @type {number} */
  lastControMouseScreen1[_0x34b6[291]] = -1;
  /** @type {number} */
  lastControMouseScreen2[_0x34b6[290]] = -1;
  /** @type {number} */
  lastControMouseScreen2[_0x34b6[291]] = -1;
  /** @type {boolean} */
  bControlStartPan = false;
  /** @type {boolean} */
  controlNaviIconMesh[_0x34b6[330]] = false;
}
/**
 * @param {?} local$$59198
 * @return {undefined}
 */
function onLSJDivMouseUp(local$$59198) {
  local$$59198[_0x34b6[1428]]();
  controlClickEnd[_0x34b6[334]](local$$59198[_0x34b6[1429]], local$$59198[_0x34b6[1430]]);
  mouseUp();
  if (local$$59198[_0x34b6[1594]] == THREE[_0x34b6[1955]][_0x34b6[1954]]) {
    doObjectClickEvent(local$$59198[_0x34b6[1429]], local$$59198[_0x34b6[1430]]);
  }
  if (local$$59198[_0x34b6[1594]] == THREE[_0x34b6[1955]][_0x34b6[1956]]) {
    doObjectClickEvent1(local$$59198[_0x34b6[1429]], local$$59198[_0x34b6[1430]]);
  }
  try {
    if (onCustomMouseUp && typeof onCustomMouseUp == _0x34b6[391]) {
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
  local$$59286[_0x34b6[290]] = local$$59281 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
  /** @type {number} */
  local$$59286[_0x34b6[291]] = -(local$$59282 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
  var local$$59323 = new THREE.Raycaster;
  local$$59323[_0x34b6[1431]](local$$59286, controlCamera);
  var local$$59341 = local$$59323[_0x34b6[1437]](controlFeatueLOD[_0x34b6[1446]][_0x34b6[684]], true);
  if (local$$59341[_0x34b6[223]] == 0) {
    local$$59341 = local$$59323[_0x34b6[1437]](controlLayers[_0x34b6[1446]][_0x34b6[684]], true);
  }
  var local$$59366 = new THREE.Vector3;
  if (local$$59341[_0x34b6[223]] > 0) {
    if (local$$59341[0] != null && local$$59341[0] != undefined) {
      local$$59366[_0x34b6[338]](local$$59341[0][_0x34b6[1435]]);
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
  local$$59406[_0x34b6[1477]](controlCamera);
  var local$$59421 = new THREE.Vector2(local$$59406[_0x34b6[290]], local$$59406[_0x34b6[291]]);
  /** @type {number} */
  local$$59421[_0x34b6[290]] = (local$$59421[_0x34b6[290]] * .5 + .5) * controlRender[_0x34b6[1694]][_0x34b6[545]];
  /** @type {number} */
  local$$59421[_0x34b6[291]] = controlRender[_0x34b6[1694]][_0x34b6[548]] - (local$$59421[_0x34b6[291]] * .5 + .5) * controlRender[_0x34b6[1694]][_0x34b6[548]];
  return local$$59421;
}
/**
 * @param {number} local$$59472
 * @param {number} local$$59473
 * @return {?}
 */
function intersectScene(local$$59472, local$$59473) {
  /** @type {number} */
  controlMouse[_0x34b6[290]] = local$$59472 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
  /** @type {number} */
  controlMouse[_0x34b6[291]] = -(local$$59473 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
  controlRaycaster[_0x34b6[1431]](controlMouse, controlCamera);
  var local$$59524 = controlRaycaster[_0x34b6[1437]](controlFeatueLOD[_0x34b6[1446]][_0x34b6[684]], true);
  if (local$$59524[_0x34b6[223]] == 0) {
    local$$59524 = controlRaycaster[_0x34b6[1437]](controlLayers[_0x34b6[1446]][_0x34b6[684]], true);
  }
  intersectsObj = controlRaycaster[_0x34b6[1437]](clippingControl[_0x34b6[1446]][_0x34b6[684]], true);
  if (local$$59524[_0x34b6[223]] > 0) {
    if (local$$59524[0]) {
      return local$$59524[0][_0x34b6[1435]];
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
  if (LSJMath[_0x34b6[1704]](local$$59587)) {
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
  var local$$59626 = (new THREE.Vector3)[_0x34b6[338]](local$$59604);
  local$$59626[_0x34b6[1434]](controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]]);
  local$$59626[_0x34b6[1957]](local$$59618);
  local$$59626[_0x34b6[1957]](local$$59611);
  return local$$59626[_0x34b6[223]]();
}
/**
 * @param {?} local$$59657
 * @return {?}
 */
function getCamXYLen(local$$59657) {
  var local$$59664 = new THREE.Vector3(0, 0, 1);
  var local$$59672 = (new THREE.Vector3)[_0x34b6[338]](local$$59657);
  local$$59672[_0x34b6[1434]](controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]]);
  local$$59672[_0x34b6[1957]](local$$59664);
  return local$$59672[_0x34b6[223]]();
}
/**
 * @param {?} local$$59698
 * @return {?}
 */
function isNewCamAltOutOfRange(local$$59698) {
  var local$$59704 = getCamAltitude(controlCamera[_0x34b6[430]]);
  var local$$59707 = getCamAltitude(local$$59698);
  if (local$$59707 < local$$59704) {
    return false;
  }
  var local$$59721 = controlFeatueLOD[_0x34b6[1759]][_0x34b6[1693]];
  local$$59721 = controlLayers[_0x34b6[1450]]()[_0x34b6[1693]] > local$$59721 ? controlLayers[_0x34b6[1450]]()[_0x34b6[1693]] : local$$59721;
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
  var local$$59762 = getCamXYLen(controlCamera[_0x34b6[430]]);
  var local$$59765 = getCamXYLen(local$$59756);
  if (local$$59765 < local$$59762) {
    return false;
  }
  var local$$59779 = controlFeatueLOD[_0x34b6[1759]][_0x34b6[1693]];
  local$$59779 = controlLayers[_0x34b6[1450]]()[_0x34b6[1693]] > local$$59779 ? controlLayers[_0x34b6[1450]]()[_0x34b6[1693]] : local$$59779;
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
  local$$59835[_0x34b6[1696]](local$$59814);
  local$$59821[_0x34b6[1696]](local$$59814);
  var local$$59851 = local$$59835[_0x34b6[1600]](local$$59828);
  var local$$59859 = (new THREE.Vector3)[_0x34b6[338]](local$$59835);
  local$$59859[_0x34b6[1599]](local$$59828)[_0x34b6[1487]]();
  var local$$59874 = local$$59859[_0x34b6[1432]](local$$59821);
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
  var local$$59893 = getCamTilt(controlCamera[_0x34b6[1239]]);
  var local$$59896 = getCamTilt(local$$59887);
  if (local$$59893 < controlMinTilt && local$$59896 > local$$59893 && local$$59896 <= Math[_0x34b6[979]] / 2) {
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
  var local$$59957 = controlFeatueLOD[_0x34b6[1759]][_0x34b6[1693]] == 0 ? 100 : controlFeatueLOD[_0x34b6[1759]][_0x34b6[1693]];
  local$$59957 = controlLayers[_0x34b6[1450]]()[_0x34b6[1693]] > local$$59957 ? controlLayers[_0x34b6[1450]]()[_0x34b6[1693]] : local$$59957;
  /** @type {number} */
  movePlaneMesh[_0x34b6[1090]][_0x34b6[290]] = local$$59957 * 10;
  /** @type {number} */
  movePlaneMesh[_0x34b6[1090]][_0x34b6[291]] = local$$59957 * 10;
  /** @type {number} */
  movePlaneMesh[_0x34b6[1090]][_0x34b6[1287]] = 1;
  movePlaneMesh[_0x34b6[430]][_0x34b6[338]](local$$59939);
  movePlaneMesh[_0x34b6[1263]](true);
  /** @type {number} */
  controlMouse2[_0x34b6[290]] = local$$59937 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
  /** @type {number} */
  controlMouse2[_0x34b6[291]] = -(local$$59938 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
  controlRaycaster2[_0x34b6[1431]](controlMouse2, controlCamera);
  var local$$60063 = controlRaycaster2[_0x34b6[1437]](movePlaneMeshs, true);
  if (local$$60063[_0x34b6[223]] < 1) {
    return new THREE.Vector3;
  }
  return local$$60063[0][_0x34b6[1435]];
}
/**
 * @param {number} local$$60084
 * @param {number} local$$60085
 * @return {?}
 */
function getPageLODCenterPlaneIntersectPos(local$$60084, local$$60085) {
  return getPlaneIntersectPos(local$$60084, local$$60085, controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]]);
}
/**
 * @param {?} local$$60098
 * @param {number} local$$60099
 * @return {undefined}
 */
function panSceneDelta(local$$60098, local$$60099) {
  var local$$60107 = (new THREE.Vector3)[_0x34b6[338]](local$$60098);
  local$$60107[_0x34b6[350]](local$$60099);
  var local$$60123 = (new THREE.Vector3)[_0x34b6[338]](controlCamera[_0x34b6[430]]);
  local$$60123[_0x34b6[1434]](local$$60107);
  if (isNewCamXYOutOfRange(local$$60123)) {
    return;
  }
  controlCamera[_0x34b6[430]][_0x34b6[1434]](local$$60107);
  controlCamera[_0x34b6[1263]]();
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
  if (LSJMath[_0x34b6[1703]](lastControMouseScreen1)) {
    lastControMouseScreen1[_0x34b6[290]] = local$$60151;
    lastControMouseScreen1[_0x34b6[291]] = local$$60152;
    return;
  }
  var local$$60184 = intersectSceneAndPlane(lastControMouseScreen1[_0x34b6[290]], lastControMouseScreen1[_0x34b6[291]]);
  if (LSJMath[_0x34b6[1704]](local$$60184)) {
    lastControMouseScreen1[_0x34b6[290]] = local$$60151;
    lastControMouseScreen1[_0x34b6[291]] = local$$60152;
    return;
  }
  lastControMouseScreen1[_0x34b6[290]] = local$$60151;
  lastControMouseScreen1[_0x34b6[291]] = local$$60152;
  var local$$60215 = getPlaneIntersectPos(local$$60151, local$$60152, local$$60184);
  if (LSJMath[_0x34b6[1704]](local$$60215)) {
    return;
  }
  var local$$60227 = new THREE.Vector3;
  local$$60227[_0x34b6[338]](local$$60215)[_0x34b6[1434]](local$$60184);
  /** @type {number} */
  local$$60227[_0x34b6[1287]] = 0;
  var local$$60253 = (new THREE.Vector3)[_0x34b6[338]](controlCamera[_0x34b6[430]]);
  local$$60253[_0x34b6[1434]](local$$60227);
  if (isNewCamXYOutOfRange(local$$60253)) {
    return;
  }
  controlCamera[_0x34b6[430]][_0x34b6[1434]](local$$60227);
  controlCamera[_0x34b6[1263]]();
  controlInertiaParam[_0x34b6[1958]][_0x34b6[338]](local$$60227);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1959]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1960]] = 0;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1961]] = 1;
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
  if (LSJMath[_0x34b6[1703]](lastControMouseScreen1)) {
    lastControMouseScreen1[_0x34b6[290]] = local$$60307;
    lastControMouseScreen1[_0x34b6[291]] = local$$60308;
    return;
  }
  var local$$60340 = intersectSceneAndPlane(lastControMouseScreen1[_0x34b6[290]], lastControMouseScreen1[_0x34b6[291]]);
  if (LSJMath[_0x34b6[1704]](local$$60340)) {
    lastControMouseScreen1[_0x34b6[290]] = local$$60307;
    lastControMouseScreen1[_0x34b6[291]] = local$$60308;
    return;
  }
  lastControMouseScreen1[_0x34b6[290]] = local$$60307;
  lastControMouseScreen1[_0x34b6[291]] = local$$60308;
  var local$$60371 = getPlaneIntersectPos(local$$60307, local$$60308, local$$60340);
  if (LSJMath[_0x34b6[1704]](local$$60371)) {
    return;
  }
  var local$$60383 = new THREE.Vector3;
  local$$60383[_0x34b6[338]](local$$60371)[_0x34b6[1434]](local$$60340);
  /** @type {number} */
  local$$60383[_0x34b6[1287]] = 0;
  var local$$60409 = (new THREE.Vector3)[_0x34b6[338]](controlCamera[_0x34b6[430]]);
  local$$60409[_0x34b6[1434]](local$$60383);
  if (isNewCamXYOutOfRange(local$$60409)) {
    return;
  }
  controlCamera[_0x34b6[430]][_0x34b6[1434]](local$$60383);
  controlCamera[_0x34b6[1263]]();
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
  var local$$60458 = (new THREE.Vector3)[_0x34b6[338]](local$$60449);
  if (LSJMath[_0x34b6[1704]](local$$60449)) {
    local$$60458[_0x34b6[338]](controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]]);
  }
  var local$$60480 = new THREE.Vector3;
  local$$60480[_0x34b6[338]](local$$60458)[_0x34b6[1434]](controlCamera[_0x34b6[430]]);
  var local$$60498 = local$$60480[_0x34b6[223]]();
  /** @type {number} */
  var local$$60507 = local$$60498 * Math[_0x34b6[1525]](1 - local$$60450);
  if (local$$60507 < controlMinZoomDist && local$$60450 > 0) {
    return new THREE.Vector3;
  }
  local$$60480[_0x34b6[350]](local$$60450);
  var local$$60533 = (new THREE.Vector3)[_0x34b6[338]](controlCamera[_0x34b6[430]]);
  local$$60533[_0x34b6[274]](local$$60480);
  if (isNewCamAltOutOfRange(local$$60533)) {
    return new THREE.Vector3;
  }
  controlCamera[_0x34b6[430]][_0x34b6[274]](local$$60480);
  controlCamera[_0x34b6[1263]]();
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
  rollSceneAngle(local$$60569, local$$60566 * Math[_0x34b6[979]] / 180, true);
}
/**
 * @param {!Object} local$$60583
 * @param {number} local$$60584
 * @param {boolean} local$$60585
 * @return {undefined}
 */
function rollSceneAngle(local$$60583, local$$60584, local$$60585) {
  var local$$60593 = (new THREE.Vector3)[_0x34b6[338]](local$$60583);
  if (LSJMath[_0x34b6[1704]](local$$60583)) {
    if (local$$60585) {
      local$$60593[_0x34b6[338]](controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]]);
    } else {
      return;
    }
  }
  var local$$60626 = (new THREE.Vector3)[_0x34b6[338]](controlCamera[_0x34b6[430]]);
  var local$$60637 = (new THREE.Quaternion)[_0x34b6[338]](controlCamera[_0x34b6[1239]]);
  var local$$60652 = (new THREE.Vector3)[_0x34b6[338]](local$$60593)[_0x34b6[1434]](controlCamera[_0x34b6[430]]);
  var local$$60667 = (new THREE.Quaternion)[_0x34b6[338]](controlCamera[_0x34b6[1239]])[_0x34b6[241]]();
  var local$$60675 = (new THREE.Vector3)[_0x34b6[338]](local$$60652);
  local$$60675[_0x34b6[1696]](local$$60667);
  var local$$60688 = (new THREE.Vector3)[_0x34b6[338]](local$$60675);
  local$$60688[_0x34b6[1696]](local$$60637);
  local$$60626[_0x34b6[274]](local$$60688);
  var local$$60713 = (new THREE.Vector3(0, 0, 1))[_0x34b6[1696]](local$$60667)[_0x34b6[1487]]();
  var local$$60717 = new THREE.Quaternion;
  local$$60717[_0x34b6[1565]](local$$60713, local$$60584);
  local$$60637[_0x34b6[336]](local$$60717);
  var local$$60735 = (new THREE.Vector3)[_0x34b6[338]](local$$60675);
  local$$60735[_0x34b6[350]](-1);
  local$$60735[_0x34b6[1696]](local$$60637);
  local$$60626[_0x34b6[274]](local$$60735);
  controlCamera[_0x34b6[1239]][_0x34b6[338]](local$$60637);
  controlCamera[_0x34b6[430]][_0x34b6[338]](local$$60626);
  controlCamera[_0x34b6[1263]]();
}
/**
 * @param {!Object} local$$60777
 * @param {number} local$$60778
 * @param {boolean} local$$60779
 * @return {undefined}
 */
function rollScene(local$$60777, local$$60778, local$$60779) {
  /** @type {number} */
  var local$$60792 = local$$60778 * Math[_0x34b6[979]] / controlRender[_0x34b6[1694]][_0x34b6[545]];
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
  pitchSceneAngle(local$$60805, local$$60802 * Math[_0x34b6[979]] / 180);
}
/**
 * @param {!Object} local$$60818
 * @param {number} local$$60819
 * @return {undefined}
 */
function pitchSceneAngle(local$$60818, local$$60819) {
  var local$$60827 = (new THREE.Vector3)[_0x34b6[338]](local$$60818);
  if (LSJMath[_0x34b6[1704]](local$$60818)) {
    local$$60827[_0x34b6[338]](controlFeatueLOD[_0x34b6[1759]][_0x34b6[658]]);
  }
  var local$$60856 = (new THREE.Vector3)[_0x34b6[338]](controlCamera[_0x34b6[430]]);
  var local$$60867 = (new THREE.Quaternion)[_0x34b6[338]](controlCamera[_0x34b6[1239]]);
  var local$$60882 = (new THREE.Vector3)[_0x34b6[338]](local$$60827)[_0x34b6[1434]](controlCamera[_0x34b6[430]]);
  var local$$60897 = (new THREE.Quaternion)[_0x34b6[338]](controlCamera[_0x34b6[1239]])[_0x34b6[241]]();
  var local$$60905 = (new THREE.Vector3)[_0x34b6[338]](local$$60882);
  local$$60905[_0x34b6[1696]](local$$60897);
  var local$$60918 = (new THREE.Vector3)[_0x34b6[338]](local$$60905);
  local$$60918[_0x34b6[1696]](local$$60867);
  local$$60856[_0x34b6[274]](local$$60918);
  var local$$60935 = new THREE.Vector3(1, 0, 0);
  var local$$60939 = new THREE.Quaternion;
  local$$60939[_0x34b6[1565]](local$$60935, local$$60819);
  local$$60867[_0x34b6[336]](local$$60939);
  var local$$60957 = (new THREE.Vector3)[_0x34b6[338]](local$$60905);
  local$$60957[_0x34b6[350]](-1);
  local$$60957[_0x34b6[1696]](local$$60867);
  local$$60856[_0x34b6[274]](local$$60957);
  if (isNewCamDirOutOfRange(local$$60867)) {
    if (controlInertiaParam[_0x34b6[1962]]) {
      /** @type {boolean} */
      controlInertiaParam[_0x34b6[1962]] = false;
    }
    return;
  }
  controlCamera[_0x34b6[1239]][_0x34b6[338]](local$$60867);
  controlCamera[_0x34b6[430]][_0x34b6[338]](local$$60856);
  controlCamera[_0x34b6[1263]]();
}
/**
 * @param {!Object} local$$61016
 * @param {number} local$$61017
 * @return {undefined}
 */
function pitchScene(local$$61016, local$$61017) {
  /** @type {number} */
  var local$$61030 = local$$61017 * Math[_0x34b6[979]] / controlRender[_0x34b6[1694]][_0x34b6[548]];
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
  controlMomentumParam[_0x34b6[1950]][_0x34b6[338]](local$$61042);
  /** @type {boolean} */
  controlMomentumParam[_0x34b6[1963]] = true;
  controlMomentumParam[_0x34b6[1964]] = local$$61039;
  zoomScene(controlMomentumParam[_0x34b6[1950]], controlMomentumParam[_0x34b6[1964]]);
}
/**
 * @param {?} local$$61074
 * @param {?} local$$61075
 * @param {?} local$$61076
 * @return {undefined}
 */
function beginMomentumRollScreen(local$$61074, local$$61075, local$$61076) {
  var local$$61079 = screenToScene(local$$61074, local$$61075);
  controlMomentumParam[_0x34b6[1965]][_0x34b6[338]](local$$61079);
  /** @type {boolean} */
  controlMomentumParam[_0x34b6[1966]] = true;
  /** @type {number} */
  controlMomentumParam[_0x34b6[1967]] = local$$61076 * Math[_0x34b6[979]] / 180;
  rollSceneAngle(controlMomentumParam[_0x34b6[1965]], controlMomentumParam[_0x34b6[1967]]);
}
/**
 * @param {?} local$$61117
 * @param {?} local$$61118
 * @param {?} local$$61119
 * @return {undefined}
 */
function beginMomentumPitchScreen(local$$61117, local$$61118, local$$61119) {
  var local$$61122 = screenToScene(local$$61117, local$$61118);
  controlMomentumParam[_0x34b6[1968]][_0x34b6[338]](local$$61122);
  /** @type {boolean} */
  controlMomentumParam[_0x34b6[1969]] = true;
  /** @type {number} */
  controlMomentumParam[_0x34b6[1970]] = local$$61119 * Math[_0x34b6[979]] / 180;
  rollSceneAngle(controlMomentumParam[_0x34b6[1968]], controlMomentumParam[_0x34b6[1970]]);
}
/**
 * @return {undefined}
 */
function stopSceneMomentum() {
  controlMomentumParam[_0x34b6[661]](LSMomentumFlag.MOMENTUM_ALL);
}
/**
 * @return {undefined}
 */
function momentumScene() {
  if (controlMomentumParam[_0x34b6[1971]]) {
    rollSceneAngle(controlMomentumParam[_0x34b6[1965]], controlMomentumParam[_0x34b6[1967]]);
  }
  if (controlMomentumParam[_0x34b6[1972]]) {
    pitchSceneAngle(controlMomentumParam[_0x34b6[1968]], controlMomentumParam[_0x34b6[1970]]);
  }
  if (controlMomentumParam[_0x34b6[1973]]) {
    zoomScene(controlMomentumParam[_0x34b6[1950]], controlMomentumParam[_0x34b6[1964]]);
  }
}
/**
 * @param {?} local$$61216
 * @param {?} local$$61217
 * @return {undefined}
 */
function inertiaPanScene(local$$61216, local$$61217) {
  if (controlInertiaParam[_0x34b6[1961]] / local$$61217 > 0) {
    controlInertiaParam[_0x34b6[1961]] += local$$61217;
  } else {
    controlInertiaParam[_0x34b6[1961]] = local$$61217;
  }
  controlInertiaParam[_0x34b6[1958]][_0x34b6[338]](local$$61216);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1959]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1960]] = 0;
}
/**
 * @param {?} local$$61262
 * @param {number} local$$61263
 * @return {undefined}
 */
function inertiaZoomScene(local$$61262, local$$61263) {
  if (controlInertiaParam[_0x34b6[1953]] / local$$61263 > 0) {
    controlInertiaParam[_0x34b6[1953]] += local$$61263;
  } else {
    /** @type {number} */
    controlInertiaParam[_0x34b6[1953]] = local$$61263;
  }
  controlInertiaParam[_0x34b6[1950]][_0x34b6[338]](local$$61262);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1951]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1952]] = 0;
}
/**
 * @param {?} local$$61308
 * @param {?} local$$61309
 * @param {?} local$$61310
 * @return {undefined}
 */
function inertiaZoomScreen(local$$61308, local$$61309, local$$61310) {
  var local$$61313 = screenToScene(local$$61308, local$$61309);
  if (cameraMode == _0x34b6[1935]) {
    local$$61313 = orbitControls[_0x34b6[1875]];
  }
  controlInertiaParam[_0x34b6[1950]][_0x34b6[338]](local$$61313);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1951]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1952]] = 0;
  controlInertiaParam[_0x34b6[1953]] = local$$61310;
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
  var local$$61366 = local$$61358 * Math[_0x34b6[979]] / 180;
  if (controlInertiaParam[_0x34b6[1974]] / local$$61366 > 0) {
    controlInertiaParam[_0x34b6[1974]] += local$$61366;
  } else {
    /** @type {number} */
    controlInertiaParam[_0x34b6[1974]] = local$$61366;
  }
  controlInertiaParam[_0x34b6[1965]][_0x34b6[338]](local$$61357);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1975]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1976]] = 0;
}
/**
 * @param {?} local$$61411
 * @param {?} local$$61412
 * @param {?} local$$61413
 * @return {undefined}
 */
function inertiaRollScreen(local$$61411, local$$61412, local$$61413) {
  var local$$61416 = screenToScene(local$$61411, local$$61412);
  controlInertiaParam[_0x34b6[1965]][_0x34b6[338]](local$$61416);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1975]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1976]] = 0;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1974]] = local$$61413 * Math[_0x34b6[979]] / 180;
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
  var local$$61464 = local$$61456 * Math[_0x34b6[979]] / 180;
  if (controlInertiaParam[_0x34b6[1977]] / local$$61464 > 0) {
    controlInertiaParam[_0x34b6[1977]] += local$$61464;
  } else {
    /** @type {number} */
    controlInertiaParam[_0x34b6[1977]] = local$$61464;
  }
  controlInertiaParam[_0x34b6[1968]][_0x34b6[338]](local$$61455);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1962]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1978]] = 0;
}
/**
 * @param {?} local$$61509
 * @param {?} local$$61510
 * @param {?} local$$61511
 * @return {undefined}
 */
function inertiaPitchScreen(local$$61509, local$$61510, local$$61511) {
  var local$$61514 = screenToScene(local$$61509, local$$61510);
  if (cameraMode == _0x34b6[1935]) {
    local$$61514 = orbitControls[_0x34b6[1875]];
  }
  controlInertiaParam[_0x34b6[1968]][_0x34b6[338]](local$$61514);
  /** @type {boolean} */
  controlInertiaParam[_0x34b6[1962]] = true;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1978]] = 0;
  /** @type {number} */
  controlInertiaParam[_0x34b6[1977]] = local$$61511 * Math[_0x34b6[979]] / 180;
  /** @type {boolean} */
  controlNeedStopInertia = true;
}
/**
 * @return {undefined}
 */
function stopSceneInertia() {
  controlInertiaParam[_0x34b6[661]](LSInertiaFlag.INERTIA_ALL);
  /** @type {boolean} */
  controlNeedStopInertia = false;
}
/**
 * @return {undefined}
 */
function inertiaScene() {
  if (controlInertiaParam[_0x34b6[1959]] && controlInertiaParam[_0x34b6[1960]] < controlInertiaParam[_0x34b6[1979]]) {
    controlInertiaParam[_0x34b6[1960]]++;
    /** @type {number} */
    var local$$61608 = controlInertiaParam[_0x34b6[1980]](controlInertiaParam[_0x34b6[1960]], controlInertiaParam[_0x34b6[1979]]) * controlInertiaParam[_0x34b6[1961]];
    panSceneDelta(controlInertiaParam[_0x34b6[1958]], local$$61608);
  }
  if (controlInertiaParam[_0x34b6[1975]] && controlInertiaParam[_0x34b6[1976]] < controlInertiaParam[_0x34b6[1981]]) {
    controlInertiaParam[_0x34b6[1976]]++;
    /** @type {number} */
    var local$$61648 = controlInertiaParam[_0x34b6[1980]](controlInertiaParam[_0x34b6[1976]], controlInertiaParam[_0x34b6[1981]]) * controlInertiaParam[_0x34b6[1974]];
    rollSceneAngle(controlInertiaParam[_0x34b6[1965]], local$$61648);
    controlInertiaParam[_0x34b6[1974]] -= local$$61648;
  }
  if (controlInertiaParam[_0x34b6[1951]] && controlInertiaParam[_0x34b6[1952]] < controlInertiaParam[_0x34b6[1982]]) {
    controlInertiaParam[_0x34b6[1952]]++;
    /** @type {number} */
    var local$$61693 = controlInertiaParam[_0x34b6[1980]](controlInertiaParam[_0x34b6[1952]], controlInertiaParam[_0x34b6[1982]]) * controlInertiaParam[_0x34b6[1953]];
    zoomScene(controlInertiaParam[_0x34b6[1950]], local$$61693);
    controlInertiaParam[_0x34b6[1953]] -= local$$61693;
  }
  if (controlInertiaParam[_0x34b6[1962]] && controlInertiaParam[_0x34b6[1978]] < controlInertiaParam[_0x34b6[1983]]) {
    controlInertiaParam[_0x34b6[1978]]++;
    /** @type {number} */
    var local$$61738 = controlInertiaParam[_0x34b6[1980]](controlInertiaParam[_0x34b6[1978]], controlInertiaParam[_0x34b6[1983]]) * controlInertiaParam[_0x34b6[1977]];
    pitchSceneAngle(controlInertiaParam[_0x34b6[1968]], local$$61738);
    controlInertiaParam[_0x34b6[1977]] -= local$$61738;
  }
}
/**
 * @param {?} local$$61755
 * @return {undefined}
 */
function onLSJDivMouseMove(local$$61755) {
  local$$61755[_0x34b6[1428]]();
  if (cameraMode == _0x34b6[1935]) {
    return;
  }
  if (local$$61755[_0x34b6[1916]] == LSJMOUSEBUTTON[_0x34b6[432]] && intersectsObj[_0x34b6[223]] == 0) {
    panSceneInertial(local$$61755[_0x34b6[1429]], local$$61755[_0x34b6[1430]]);
  } else {
    if (local$$61755[_0x34b6[1916]] == LSJMOUSEBUTTON[_0x34b6[655]] || local$$61755[_0x34b6[1916]] == LSJMOUSEBUTTON[_0x34b6[573]]) {
      if (LSJMath[_0x34b6[1703]](lastControMouseScreen1)) {
        lastControMouseScreen1[_0x34b6[290]] = local$$61755[_0x34b6[1429]];
        lastControMouseScreen1[_0x34b6[291]] = local$$61755[_0x34b6[1430]];
        return;
      }
      /** @type {number} */
      var local$$61837 = local$$61755[_0x34b6[1429]] - lastControMouseScreen1[_0x34b6[290]];
      /** @type {number} */
      var local$$61846 = local$$61755[_0x34b6[1430]] - lastControMouseScreen1[_0x34b6[291]];
      lastControMouseScreen1[_0x34b6[290]] = local$$61755[_0x34b6[1429]];
      lastControMouseScreen1[_0x34b6[291]] = local$$61755[_0x34b6[1430]];
      if (Math[_0x34b6[1525]](local$$61837) > Math[_0x34b6[1525]](local$$61846)) {
        if (controlInertiaUsed) {
          /** @type {number} */
          var local$$61882 = local$$61837 * 360 / controlRender[_0x34b6[1694]][_0x34b6[545]];
          inertiaRollScene(mouseDWorldPos1, -local$$61882);
        } else {
          rollScene(mouseDWorldPos1, -local$$61837, true);
        }
      } else {
        if (controlInertiaUsed) {
          /** @type {number} */
          local$$61882 = local$$61846 * 180 / controlRender[_0x34b6[1694]][_0x34b6[548]];
          inertiaPitchScene(mouseDWorldPos1, local$$61882);
        } else {
          pitchScene(mouseDWorldPos1, local$$61846);
        }
      }
    }
  }
  try {
    if (onCustomMouseMove && typeof onCustomMouseMove == _0x34b6[391]) {
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
  local$$61939[_0x34b6[1428]]();
  switch(local$$61939[_0x34b6[1984]][_0x34b6[223]]) {
    case 1:
      mouseDown(local$$61939[_0x34b6[1984]][0][_0x34b6[1429]], local$$61939[_0x34b6[1984]][0][_0x34b6[1430]], LSJMOUSEBUTTON[_0x34b6[432]]);
      break;
    case 2:
      doubleTouchStart(local$$61939[_0x34b6[1984]][0][_0x34b6[1429]], local$$61939[_0x34b6[1984]][0][_0x34b6[1430]], local$$61939[_0x34b6[1984]][1][_0x34b6[1429]], local$$61939[_0x34b6[1984]][1][_0x34b6[1430]]);
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
  controlClickEnd[_0x34b6[334]](local$$62022[_0x34b6[1595]][0][_0x34b6[1429]], local$$62022[_0x34b6[1595]][0][_0x34b6[1430]]);
  mouseUp();
  doObjectClickEvent(controlClickEnd[_0x34b6[290]], controlClickEnd[_0x34b6[291]]);
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
  return Math[_0x34b6[889]](local$$62076 * local$$62076 + local$$62079 * local$$62079);
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
    local$$62102[_0x34b6[290]] = -(local$$62114 - local$$62123) / (local$$62110 - local$$62119);
    /** @type {number} */
    local$$62102[_0x34b6[291]] = (local$$62110 * local$$62123 - local$$62119 * local$$62114) / (local$$62110 - local$$62119);
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
      local$$62102[_0x34b6[290]] = local$$62092;
      /** @type {number} */
      local$$62102[_0x34b6[291]] = local$$62155 * local$$62102[_0x34b6[290]] + local$$62157;
    } else {
      if (local$$62092 != local$$62094) {
        /** @type {number} */
        local$$62155 = (local$$62093 - local$$62095) / (local$$62092 - local$$62094);
        /** @type {number} */
        local$$62157 = local$$62093 - local$$62155 * local$$62092;
        local$$62102 = new THREE.Vector2;
        /** @type {number} */
        local$$62102[_0x34b6[290]] = local$$62096;
        /** @type {number} */
        local$$62102[_0x34b6[291]] = local$$62155 * local$$62102[_0x34b6[290]] + local$$62157;
      }
    }
  }
  if (local$$62102) {
    if (local$$62102[_0x34b6[290]] < Math[_0x34b6[472]](local$$62092, local$$62094) || local$$62102[_0x34b6[290]] > Math[_0x34b6[532]](local$$62092, local$$62094) || local$$62102[_0x34b6[290]] < Math[_0x34b6[472]](local$$62096, local$$62098) || local$$62102[_0x34b6[290]] > Math[_0x34b6[532]](local$$62096, local$$62098) || local$$62102[_0x34b6[291]] < Math[_0x34b6[472]](local$$62093, local$$62095) || local$$62102[_0x34b6[291]] > Math[_0x34b6[532]](local$$62093, local$$62095) || local$$62102[_0x34b6[291]] < 
    Math[_0x34b6[472]](local$$62097, local$$62099) || local$$62102[_0x34b6[291]] > Math[_0x34b6[532]](local$$62097, local$$62099)) {
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
  local$$62323[_0x34b6[1428]]();
  local$$62323[_0x34b6[1596]]();
  if (cameraMode == _0x34b6[1935]) {
    return;
  }
  switch(local$$62323[_0x34b6[1984]][_0x34b6[223]]) {
    case 1:
      panSceneInertial(local$$62323[_0x34b6[1984]][0][_0x34b6[1429]], local$$62323[_0x34b6[1984]][0][_0x34b6[1430]]);
      lastControMouseScreen1[_0x34b6[334]](local$$62323[_0x34b6[1984]][0][_0x34b6[1429]], local$$62323[_0x34b6[1984]][0][_0x34b6[1430]]);
      break;
    case 2:
      var local$$62400 = local$$62323[_0x34b6[1984]][0][_0x34b6[1429]];
      var local$$62410 = local$$62323[_0x34b6[1984]][0][_0x34b6[1430]];
      var local$$62420 = local$$62323[_0x34b6[1984]][1][_0x34b6[1429]];
      var local$$62430 = local$$62323[_0x34b6[1984]][1][_0x34b6[1430]];
      if (lastControMouseScreen1[_0x34b6[290]] >= 0 && lastControMouseScreen2[_0x34b6[290]] >= 0) {
        var local$$62450 = getDist(lastControMouseScreen1[_0x34b6[290]], lastControMouseScreen1[_0x34b6[291]], local$$62400, local$$62410);
        var local$$62459 = getDist(lastControMouseScreen2[_0x34b6[290]], lastControMouseScreen2[_0x34b6[291]], local$$62420, local$$62430);
        var local$$62474 = getDist(lastControMouseScreen1[_0x34b6[290]], lastControMouseScreen1[_0x34b6[291]], lastControMouseScreen2[_0x34b6[290]], lastControMouseScreen2[_0x34b6[291]]);
        var local$$62477 = getDist(local$$62400, local$$62410, local$$62420, local$$62430);
        var local$$62491 = Math[_0x34b6[1985]]((lastControMouseScreen1[_0x34b6[290]] - lastControMouseScreen2[_0x34b6[290]]) / local$$62474);
        var local$$62499 = Math[_0x34b6[1985]]((local$$62400 - local$$62420) / local$$62477);
        if (local$$62459 >= 0 || local$$62450 >= 0) {
          var local$$62514 = Math[_0x34b6[1525]](local$$62400 - lastControMouseScreen1[_0x34b6[290]]);
          var local$$62524 = Math[_0x34b6[1525]](local$$62410 - lastControMouseScreen1[_0x34b6[291]]);
          var local$$62534 = Math[_0x34b6[1525]](local$$62420 - lastControMouseScreen2[_0x34b6[290]]);
          var local$$62544 = Math[_0x34b6[1525]](local$$62430 - lastControMouseScreen2[_0x34b6[291]]);
          /** @type {number} */
          var local$$62547 = 0;
          if (local$$62400 < local$$62420 && local$$62410 < local$$62430 || local$$62400 > local$$62420 && local$$62410 < local$$62430) {
            /** @type {number} */
            local$$62547 = local$$62491 - local$$62499;
          } else {
            /** @type {number} */
            local$$62547 = -(local$$62491 - local$$62499);
          }
          if (Math[_0x34b6[1525]](local$$62547) < .01) {
            /** @type {number} */
            var local$$62577 = local$$62410 - lastControMouseScreen1[_0x34b6[291]];
            /** @type {number} */
            var local$$62583 = local$$62430 - lastControMouseScreen2[_0x34b6[291]];
            /** @type {number} */
            var local$$62589 = local$$62400 - lastControMouseScreen1[_0x34b6[290]];
            /** @type {number} */
            var local$$62595 = local$$62420 - lastControMouseScreen2[_0x34b6[290]];
            if (local$$62589 * local$$62595 > 0) {
              /** @type {number} */
              local$$62595 = local$$62595 == 0 ? 1 : local$$62595;
              /** @type {number} */
              var local$$62607 = local$$62589 / local$$62595;
              rollSceneAngle(touchPichPos, local$$62607 * Math[_0x34b6[472]](local$$62589, local$$62595) * .01, true);
            }
          } else {
            rollSceneAngle(touchPichPos, local$$62547, true);
          }
          if (local$$62524 > local$$62514 && local$$62544 > local$$62534 && local$$62524 > 3 && local$$62544 > 3 && local$$62577 * local$$62583 > 0 && Math[_0x34b6[1525]](local$$62474 - local$$62477) < 5) {
            /** @type {number} */
            var local$$62650 = local$$62577 / local$$62524;
            pitchScene(touchPichPos, local$$62650 * Math[_0x34b6[472]](local$$62524, local$$62544));
          }
          var local$$62678 = Math[_0x34b6[472]](controlRender[_0x34b6[1694]][_0x34b6[548]], controlRender[_0x34b6[1694]][_0x34b6[545]]);
          /** @type {number} */
          var local$$62684 = 2 * (local$$62477 - local$$62474) / local$$62678;
          zoomSceneScreen((local$$62400 + local$$62420) / 2, (local$$62410 + local$$62430) / 2, local$$62684);
        }
      }
      lastControMouseScreen1[_0x34b6[334]](local$$62400, local$$62410);
      lastControMouseScreen2[_0x34b6[334]](local$$62420, local$$62430);
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
  if (local$$62719[_0x34b6[445]] == _0x34b6[1727]) {
    local$$62719[_0x34b6[1729]][_0x34b6[1986]](local$$62719);
    outlinePass[_0x34b6[280]][_0x34b6[220]](local$$62719[_0x34b6[1446]]);
  }
}
/**
 * @return {undefined}
 */
function releaseSelectedObject() {
  getLayers()[_0x34b6[1457]]();
  /** @type {!Array} */
  outlinePass[_0x34b6[280]] = [];
}
/**
 * @param {?} local$$62766
 * @param {?} local$$62767
 * @return {undefined}
 */
function doObjectClickEvent(local$$62766, local$$62767) {
  var local$$62780 = Math[_0x34b6[1525]](controlClickEnd[_0x34b6[290]] - controlClickStart[_0x34b6[290]]);
  var local$$62793 = Math[_0x34b6[1525]](controlClickEnd[_0x34b6[291]] - controlClickStart[_0x34b6[291]]);
  if (local$$62780 < 2 && local$$62793 < 2) {
    if (onControlPageLODClick != undefined) {
      var local$$62804 = new THREE.Vector2;
      /** @type {number} */
      local$$62804[_0x34b6[290]] = local$$62766 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
      /** @type {number} */
      local$$62804[_0x34b6[291]] = -(local$$62767 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
      var local$$62841 = new THREE.Raycaster;
      local$$62841[_0x34b6[1431]](local$$62804, controlCamera);
      var local$$62859 = local$$62841[_0x34b6[1437]](controlFeatueLOD[_0x34b6[1446]][_0x34b6[684]], true);
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
      if (local$$62859[_0x34b6[223]] > 0) {
        local$$62868 = local$$62859[0][_0x34b6[1435]][_0x34b6[290]];
        local$$62871 = local$$62859[0][_0x34b6[1435]][_0x34b6[291]];
        local$$62874 = local$$62859[0][_0x34b6[1435]][_0x34b6[1287]];
        local$$62865 = local$$62859[0][_0x34b6[368]];
        onControlPageLODClick(new THREE.Vector2(local$$62766, local$$62767), new THREE.Vector3(local$$62868, local$$62871, local$$62874));
      }
    }
    if (onControlFeatureClick != undefined) {
      local$$62804 = new THREE.Vector2;
      /** @type {number} */
      local$$62804[_0x34b6[290]] = local$$62766 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
      /** @type {number} */
      local$$62804[_0x34b6[291]] = -(local$$62767 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
      local$$62841 = new THREE.Raycaster;
      local$$62841[_0x34b6[1431]](local$$62804, controlCamera);
      local$$62859 = local$$62841[_0x34b6[1437]](controlLayers[_0x34b6[1446]][_0x34b6[684]], true);
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
      if (local$$62859[_0x34b6[223]] > 0) {
        local$$62868 = local$$62859[0][_0x34b6[1435]][_0x34b6[290]];
        local$$62871 = local$$62859[0][_0x34b6[1435]][_0x34b6[291]];
        local$$62874 = local$$62859[0][_0x34b6[1435]][_0x34b6[1287]];
        local$$62865 = local$$62859[0][_0x34b6[368]];
        if (local$$62865 != null && local$$62865 != undefined) {
          var local$$63056 = local$$62865[_0x34b6[667]];
          for (; local$$63056 != undefined && local$$63056[_0x34b6[445]] == _0x34b6[1987] && local$$63056[_0x34b6[1988]] == undefined;) {
            local$$63056 = local$$63056[_0x34b6[667]];
          }
          if (local$$63056 != undefined && local$$63056[_0x34b6[1988]] != undefined && local$$63056[_0x34b6[1988]][_0x34b6[445]] == _0x34b6[1727]) {
            local$$63056[_0x34b6[1988]][_0x34b6[1729]][_0x34b6[1986]](local$$63056.Owner);
            outlinePass[_0x34b6[280]][_0x34b6[220]](local$$63056);
            onControlFeatureClick(new THREE.Vector2(local$$62766, local$$62767), new THREE.Vector3(local$$62868, local$$62871, local$$62874), local$$63056.Owner);
          } else {
            if (local$$62865[_0x34b6[445]] != _0x34b6[1290] && local$$62865[_0x34b6[445]] != _0x34b6[1682]) {
              outlinePass[_0x34b6[280]][_0x34b6[220]](local$$62859[0][_0x34b6[368]]);
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
  var local$$63186 = Math[_0x34b6[1525]](controlClickEnd[_0x34b6[290]] - controlClickStart[_0x34b6[290]]);
  var local$$63199 = Math[_0x34b6[1525]](controlClickEnd[_0x34b6[291]] - controlClickStart[_0x34b6[291]]);
  if (local$$63186 < 2 && local$$63199 < 2) {
    if (onControlPageLODRightClick != undefined) {
      var local$$63210 = new THREE.Vector2;
      /** @type {number} */
      local$$63210[_0x34b6[290]] = local$$63172 / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
      /** @type {number} */
      local$$63210[_0x34b6[291]] = -(local$$63173 / controlRender[_0x34b6[1694]][_0x34b6[548]]) * 2 + 1;
      var local$$63247 = new THREE.Raycaster;
      local$$63247[_0x34b6[1431]](local$$63210, controlCamera);
      var local$$63265 = local$$63247[_0x34b6[1437]](controlFeatueLOD[_0x34b6[1446]][_0x34b6[684]], true);
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
      if (local$$63265[_0x34b6[223]] > 0) {
        local$$63274 = local$$63265[0][_0x34b6[1435]][_0x34b6[290]];
        local$$63277 = local$$63265[0][_0x34b6[1435]][_0x34b6[291]];
        local$$63280 = local$$63265[0][_0x34b6[1435]][_0x34b6[1287]];
        local$$63271 = local$$63265[0][_0x34b6[368]];
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
  flyToCameraControls[_0x34b6[661]]();
  flyWithLineControls[_0x34b6[1586]]();
  flyAroundCenterControls[_0x34b6[661]]();
  local$$63339[_0x34b6[1428]]();
  if (cameraMode == _0x34b6[1935]) {
    return;
  }
  /** @type {number} */
  var local$$63369 = 0;
  if (local$$63339[_0x34b6[1989]]) {
    /** @type {number} */
    local$$63369 = local$$63339[_0x34b6[1989]] / 40;
  } else {
    if (local$$63339[_0x34b6[1990]]) {
      /** @type {number} */
      local$$63369 = -local$$63339[_0x34b6[1990]] / 3;
    }
  }
  /** @type {number} */
  var local$$63400 = local$$63369 / 30;
  if (controlNeedStopInertia) {
    stopSceneInertia();
  }
  var local$$63414 = intersectSceneAndPlane(local$$63339[_0x34b6[1429]], local$$63339[_0x34b6[1430]]);
  if (!LSJMath[_0x34b6[1704]](local$$63414)) {
    if (controlInertiaUsed) {
      inertiaZoomScene(local$$63414, local$$63400);
    } else {
      zoomScene(local$$63414, local$$63400);
    }
    updateNaviIconMesh(local$$63414);
    controlNaviLiveTime = Date[_0x34b6[348]]();
  }
}
/**
 * @param {?} local$$63442
 * @return {undefined}
 */
function onLSJContextmenu(local$$63442) {
  local$$63442[_0x34b6[1428]]();
}
/**
 * @return {undefined}
 */
function activateSceneControlMouseEvent() {
  controlDiv[_0x34b6[1423]](_0x34b6[1422], onLSJDivMouseMove, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1424], onLSJDivMouseDown, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1425], onLSJDivMouseUp, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1991], onLSJDivMouseWheel, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1992], onLSJContextmenu, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1993], onLSJDivMouseWheel, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1579], onLSJDivTouchStart, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1582], onLSJDivTouchEnd, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1583], onLSJDivTouchCancel, false);
  controlDiv[_0x34b6[1423]](_0x34b6[1580], onLSJDivTouchMove, false);
}
/**
 * @return {undefined}
 */
function dectivateSceneControlMouseEvent() {
  controlDiv[_0x34b6[1427]](_0x34b6[1422], onLSJDivMouseMove, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1424], onLSJDivMouseDown, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1425], onLSJDivMouseUp, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1991], onLSJDivMouseWheel, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1992], onLSJContextmenu, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1993], onLSJDivMouseWheel, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1579], onLSJDivTouchStart, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1582], onLSJDivTouchEnd, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1583], onLSJDivTouchCancel, false);
  controlDiv[_0x34b6[1427]](_0x34b6[1580], onLSJDivTouchMove, false);
}
var DisplayMode = {
  None : 0,
  Wireframe : 1,
  Heightmap : 2,
  HeightmapWireframe : 3
};
var displayMode = DisplayMode[_0x34b6[1994]];
/**
 * @param {?} local$$63628
 * @param {!Object} local$$63629
 * @return {undefined}
 */
var setDisplayMode = function(local$$63628, local$$63629) {
  switch(local$$63628) {
    case DisplayMode[_0x34b6[1994]]:
      displayMode = DisplayMode[_0x34b6[1994]];
      /** @type {null} */
      controlScene[_0x34b6[255]] = null;
      break;
    case DisplayMode[_0x34b6[1854]]:
      displayMode = DisplayMode[_0x34b6[1854]];
      /** @type {null} */
      controlScene[_0x34b6[255]] = null;
      break;
    case DisplayMode[_0x34b6[1995]]:
      {
        displayMode = DisplayMode[_0x34b6[1995]];
        var local$$63690 = _0x34b6[1996] + _0x34b6[1997] + _0x34b6[1998] + _0x34b6[6] + _0x34b6[1999] + _0x34b6[7];
        var local$$63727 = _0x34b6[2E3] + _0x34b6[1] + _0x34b6[2001] + _0x34b6[2002] + _0x34b6[2003] + _0x34b6[2004] + _0x34b6[2005] + _0x34b6[2006] + _0x34b6[2007] + _0x34b6[2008] + _0x34b6[2009] + _0x34b6[2010];
        var local$$63732 = {
          colorRange : {
            value : local$$63629
          }
        };
        controlScene[_0x34b6[255]] = new THREE.ShaderMaterial({
          uniforms : local$$63732,
          vertexShader : local$$63690,
          fragmentShader : local$$63727
        });
        break;
      }
    case DisplayMode[_0x34b6[2011]]:
      {
        displayMode = DisplayMode[_0x34b6[2011]];
        local$$63690 = _0x34b6[1996] + _0x34b6[1997] + _0x34b6[1998] + _0x34b6[6] + _0x34b6[1999] + _0x34b6[7];
        local$$63727 = _0x34b6[2E3] + _0x34b6[1] + _0x34b6[2001] + _0x34b6[2002] + _0x34b6[2003] + _0x34b6[2004] + _0x34b6[2005] + _0x34b6[2006] + _0x34b6[2007] + _0x34b6[2008] + _0x34b6[2009] + _0x34b6[2010];
        local$$63732 = {
          colorRange : {
            value : local$$63629
          }
        };
        controlScene[_0x34b6[255]] = new THREE.ShaderMaterial({
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
  this[_0x34b6[1971]] = false;
  /** @type {boolean} */
  this[_0x34b6[1972]] = false;
  /** @type {boolean} */
  this[_0x34b6[1973]] = false;
  this[_0x34b6[1965]] = new THREE.Vector3;
  this[_0x34b6[1968]] = new THREE.Vector3;
  this[_0x34b6[1950]] = new THREE.Vector3;
  /** @type {number} */
  this[_0x34b6[1967]] = 0;
  /** @type {number} */
  this[_0x34b6[1970]] = 0;
  /** @type {number} */
  this[_0x34b6[1964]] = 0;
};
/** @type {function(): undefined} */
LSJMomentumParam[_0x34b6[219]][_0x34b6[1183]] = LSJMomentumParam;
/**
 * @param {?} local$$63930
 * @return {?}
 */
LSJMomentumParam[_0x34b6[219]][_0x34b6[2012]] = function(local$$63930) {
  if (local$$63930 & LSMomentumFlag[_0x34b6[2013]]) {
    return this[_0x34b6[1967]];
  }
  if (local$$63930 & LSMomentumFlag[_0x34b6[2014]]) {
    return this[_0x34b6[1970]];
  }
  if (local$$63930 & LSMomentumFlag[_0x34b6[2015]]) {
    return this[_0x34b6[1964]];
  }
  return 0;
};
/**
 * @param {?} local$$63977
 * @return {undefined}
 */
LSJMomentumParam[_0x34b6[219]][_0x34b6[495]] = function(local$$63977) {
  if (local$$63977 & LSMomentumFlag[_0x34b6[2013]]) {
    /** @type {boolean} */
    this[_0x34b6[1971]] = true;
  }
  if (local$$63977 & LSMomentumFlag[_0x34b6[2014]]) {
    /** @type {boolean} */
    this[_0x34b6[1972]] = true;
  }
  if (local$$63977 & LSMomentumFlag[_0x34b6[2015]]) {
    /** @type {boolean} */
    this[_0x34b6[1973]] = true;
  }
};
/**
 * @param {?} local$$64027
 * @return {undefined}
 */
LSJMomentumParam[_0x34b6[219]][_0x34b6[661]] = function(local$$64027) {
  if (local$$64027 & LSMomentumFlag[_0x34b6[2013]]) {
    /** @type {boolean} */
    this[_0x34b6[1971]] = false;
    /** @type {number} */
    this[_0x34b6[1967]] = 0;
  }
  if (local$$64027 & LSMomentumFlag[_0x34b6[2014]]) {
    /** @type {boolean} */
    this[_0x34b6[1972]] = false;
    /** @type {number} */
    this[_0x34b6[1970]] = 0;
  }
  if (local$$64027 & LSMomentumFlag[_0x34b6[2015]]) {
    /** @type {boolean} */
    this[_0x34b6[1973]] = false;
    /** @type {number} */
    this[_0x34b6[1964]] = 0;
  }
};
/**
 * @return {undefined}
 */
LSJInertiaParam = function() {
  /** @type {boolean} */
  this[_0x34b6[1962]] = false;
  /** @type {boolean} */
  this[_0x34b6[1951]] = false;
  /** @type {boolean} */
  this[_0x34b6[1975]] = false;
  /** @type {boolean} */
  this[_0x34b6[1959]] = false;
  /** @type {number} */
  this[_0x34b6[1978]] = 0;
  /** @type {number} */
  this[_0x34b6[1952]] = 0;
  /** @type {number} */
  this[_0x34b6[1976]] = 0;
  /** @type {number} */
  this[_0x34b6[1960]] = 0;
  /** @type {number} */
  this[_0x34b6[1979]] = 20;
  /** @type {number} */
  this[_0x34b6[1982]] = 30;
  /** @type {number} */
  this[_0x34b6[1981]] = 20;
  /** @type {number} */
  this[_0x34b6[1983]] = 10;
  /** @type {number} */
  this[_0x34b6[1977]] = 0;
  /** @type {number} */
  this[_0x34b6[1974]] = 0;
  /** @type {number} */
  this[_0x34b6[1953]] = 0;
  /** @type {number} */
  this[_0x34b6[1961]] = 0;
  this[_0x34b6[1965]] = new THREE.Vector3;
  this[_0x34b6[1968]] = new THREE.Vector3;
  this[_0x34b6[1950]] = new THREE.Vector3;
  this[_0x34b6[1958]] = new THREE.Vector3;
};
/** @type {function(): undefined} */
LSJInertiaParam[_0x34b6[219]][_0x34b6[1183]] = LSJInertiaParam;
/**
 * @param {?} local$$64232
 * @param {number} local$$64233
 * @return {?}
 */
LSJInertiaParam[_0x34b6[219]][_0x34b6[1980]] = function(local$$64232, local$$64233) {
  return (2 * (local$$64233 + 1 - local$$64232) - 1) / (local$$64233 * local$$64233);
};
/**
 * @param {?} local$$64255
 * @return {undefined}
 */
LSJInertiaParam[_0x34b6[219]][_0x34b6[495]] = function(local$$64255) {
  if (local$$64255 & LSInertiaFlag[_0x34b6[2016]]) {
    /** @type {boolean} */
    this[_0x34b6[1959]] = true;
  }
  if (local$$64255 & LSInertiaFlag[_0x34b6[2017]]) {
    /** @type {boolean} */
    this[_0x34b6[2018]] = true;
  }
  if (local$$64255 & LSInertiaFlag[_0x34b6[2019]]) {
    /** @type {boolean} */
    this[_0x34b6[1962]] = true;
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
LSJInertiaParam[_0x34b6[219]][_0x34b6[661]] = function(local$$64318) {
  if (local$$64318 & LSInertiaFlag[_0x34b6[2016]]) {
    /** @type {boolean} */
    this[_0x34b6[1959]] = false;
    /** @type {number} */
    this[_0x34b6[1960]] = 0;
    /** @type {number} */
    this[_0x34b6[1961]] = 0;
  }
  if (local$$64318 & LSInertiaFlag[_0x34b6[2017]]) {
    /** @type {boolean} */
    this[_0x34b6[2018]] = false;
    /** @type {number} */
    this[_0x34b6[1976]] = 0;
    /** @type {number} */
    this[_0x34b6[1974]] = 0;
  }
  if (local$$64318 & LSInertiaFlag[_0x34b6[2019]]) {
    /** @type {boolean} */
    this[_0x34b6[1962]] = false;
    /** @type {number} */
    this[_0x34b6[1978]] = 0;
    /** @type {number} */
    this[_0x34b6[1977]] = 0;
  }
  if (local$$64318 & LSMomentumFlag[_0x34b6[2020]]) {
    /** @type {boolean} */
    this[_0x34b6[2021]] = false;
    /** @type {number} */
    this[_0x34b6[1952]] = 0;
    /** @type {number} */
    this[_0x34b6[1953]] = 0;
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
  this[_0x34b6[1523]] = undefined;
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {boolean} */
  this[_0x34b6[2029]] = true;
  /** @type {!Array} */
  this[_0x34b6[1906]] = [];
  /** @type {!Array} */
  this[_0x34b6[1907]] = [];
  /** @type {!Array} */
  this[_0x34b6[2030]] = [];
  this[_0x34b6[1908]] = new THREE.SphereGeometry(.4, 5, 5);
  this[_0x34b6[245]] = new THREE.Color(16711680);
};
/** @type {function(): undefined} */
LSJRulerDistance[_0x34b6[219]][_0x34b6[1183]] = LSJRulerDistance;
/**
 * @param {?} local$$64533
 * @return {undefined}
 */
LSJRulerDistance[_0x34b6[219]][_0x34b6[1909]] = function(local$$64533) {
  if (this[_0x34b6[1906]][_0x34b6[223]] < 2) {
    this[_0x34b6[1906]][_0x34b6[220]](local$$64533[_0x34b6[212]]());
    /** @type {boolean} */
    this[_0x34b6[2029]] = true;
  }
};
/**
 * @param {?} local$$64573
 * @param {?} local$$64574
 * @return {undefined}
 */
LSJRulerDistance[_0x34b6[219]][_0x34b6[1910]] = function(local$$64573, local$$64574) {
  if (this[_0x34b6[1906]][_0x34b6[223]] > local$$64573) {
    this[_0x34b6[1906]][local$$64573][_0x34b6[338]](local$$64574);
    /** @type {boolean} */
    this[_0x34b6[2029]] = true;
  }
};
/**
 * @return {undefined}
 */
LSJRulerDistance[_0x34b6[219]][_0x34b6[235]] = function() {
  /** @type {!Array} */
  this[_0x34b6[1906]] = [];
  getScene()[_0x34b6[1448]](this[_0x34b6[1446]]);
  this[_0x34b6[2022]] = undefined;
  this[_0x34b6[2023]] = undefined;
  this[_0x34b6[2024]] = undefined;
  this[_0x34b6[2025]] = undefined;
  this[_0x34b6[2026]] = undefined;
  this[_0x34b6[2027]] = undefined;
  this[_0x34b6[1523]] = undefined;
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
LSJRulerDistance[_0x34b6[219]][_0x34b6[1261]] = function(local$$64690) {
  if (this[_0x34b6[1906]][_0x34b6[223]] == 2) {
    local$$64690[_0x34b6[1448]](this[_0x34b6[1446]]);
    /** @type {!Array} */
    this[_0x34b6[2030]] = [];
    /** @type {!Array} */
    this[_0x34b6[1907]] = [];
    this[_0x34b6[1446]] = new THREE.Group;
    var local$$64732 = this[_0x34b6[1906]][0];
    var local$$64739 = this[_0x34b6[1906]][1];
    var local$$64743 = new THREE.Vector3;
    if (local$$64732[_0x34b6[1287]] < local$$64739[_0x34b6[1287]]) {
      local$$64743[_0x34b6[290]] = local$$64732[_0x34b6[290]];
      local$$64743[_0x34b6[291]] = local$$64732[_0x34b6[291]];
      local$$64743[_0x34b6[1287]] = local$$64739[_0x34b6[1287]];
    } else {
      local$$64743[_0x34b6[290]] = local$$64739[_0x34b6[290]];
      local$$64743[_0x34b6[291]] = local$$64739[_0x34b6[291]];
      local$$64743[_0x34b6[1287]] = local$$64732[_0x34b6[1287]];
    }
    var local$$64806 = new THREE.Geometry;
    local$$64806[_0x34b6[1125]][_0x34b6[220]](local$$64732[_0x34b6[212]]());
    local$$64806[_0x34b6[1125]][_0x34b6[220]](local$$64743);
    local$$64806[_0x34b6[1125]][_0x34b6[220]](local$$64743);
    local$$64806[_0x34b6[1125]][_0x34b6[220]](local$$64739[_0x34b6[212]]());
    local$$64806[_0x34b6[2031]]();
    this[_0x34b6[2028]] = new THREE.LineDashedMaterial({
      color : 65280,
      dashSize : .5,
      gapSize : .5,
      linewidth : 2
    });
    this[_0x34b6[1523]] = new THREE.LineSegments(local$$64806, this[_0x34b6[2028]]);
    this[_0x34b6[1446]][_0x34b6[274]](this.LineSegments);
    var local$$64889 = new THREE.Geometry;
    local$$64889[_0x34b6[1125]][_0x34b6[220]](local$$64732[_0x34b6[212]]());
    local$$64889[_0x34b6[1125]][_0x34b6[220]](local$$64739[_0x34b6[212]]());
    var local$$64920 = new THREE.LineBasicMaterial({
      color : 65280,
      linewidth : 2
    });
    var local$$64924 = new THREE.Line(local$$64889, local$$64920);
    this[_0x34b6[1446]][_0x34b6[274]](local$$64924);
    var local$$64936 = new THREE.Geometry;
    local$$64936[_0x34b6[1125]][_0x34b6[220]](local$$64732[_0x34b6[212]]());
    local$$64936[_0x34b6[1125]][_0x34b6[220]](local$$64743);
    local$$64936[_0x34b6[1125]][_0x34b6[220]](local$$64739[_0x34b6[212]]());
    var local$$64976 = new THREE.PointsMaterial({
      color : 16711680,
      size : 1,
      depthTest : false
    });
    var local$$64980 = new THREE.Points(local$$64936, local$$64976);
    var local$$64988 = new THREE.Mesh(this[_0x34b6[1908]], createSphereMaterial());
    local$$64988[_0x34b6[430]][_0x34b6[338]](local$$64732);
    var local$$65010 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$64988[_0x34b6[1912]]());
    var local$$65031 = projectedRadius(1, getCamera()[_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$65010, getRenderer()[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    var local$$65035 = 10 / local$$65031;
    local$$64988[_0x34b6[1090]][_0x34b6[334]](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[1907]][_0x34b6[220]](local$$64988);
    this[_0x34b6[1446]][_0x34b6[274]](local$$64988);
    var local$$65067 = new THREE.Mesh(this[_0x34b6[1908]], createSphereMaterial());
    local$$65067[_0x34b6[430]][_0x34b6[338]](local$$64743);
    local$$65067[_0x34b6[1090]][_0x34b6[334]](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[1907]][_0x34b6[220]](local$$65067);
    this[_0x34b6[1446]][_0x34b6[274]](local$$65067);
    var local$$65107 = new THREE.Mesh(this[_0x34b6[1908]], createSphereMaterial());
    local$$65107[_0x34b6[430]][_0x34b6[338]](local$$64739);
    local$$65107[_0x34b6[1090]][_0x34b6[334]](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[1907]][_0x34b6[220]](local$$65107);
    this[_0x34b6[1446]][_0x34b6[274]](local$$65107);
    this[_0x34b6[2025]] = new LSJGeoMarker;
    var local$$65148 = new LSJMarkerStyle;
    /** @type {number} */
    local$$65148[_0x34b6[1659]] = 25;
    var local$$65160 = local$$65148[_0x34b6[1672]]();
    local$$65160[_0x34b6[1652]](_0x34b6[2032]);
    local$$65160[_0x34b6[1406]]()[_0x34b6[1533]](1, 1, 1);
    local$$65160[_0x34b6[1656]](20);
    this[_0x34b6[2025]][_0x34b6[1670]](local$$65148);
    local$$65010 = local$$64739[_0x34b6[212]]()[_0x34b6[1434]](local$$64732)[_0x34b6[223]]();
    this[_0x34b6[2025]][_0x34b6[1689]](local$$65010[_0x34b6[2033]](2) + _0x34b6[2034]);
    this[_0x34b6[2025]][_0x34b6[1241]](local$$64732[_0x34b6[290]] + (local$$64739[_0x34b6[290]] - local$$64732[_0x34b6[290]]) / 2, local$$64732[_0x34b6[291]] + (local$$64739[_0x34b6[291]] - local$$64732[_0x34b6[291]]) / 2, local$$64732[_0x34b6[1287]] + (local$$64739[_0x34b6[1287]] - local$$64732[_0x34b6[1287]]) / 2);
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
    local$$65274[_0x34b6[268]][_0x34b6[1307]] = false;
    local$$65274[_0x34b6[1263]]();
    /** @type {boolean} */
    local$$65274[_0x34b6[330]] = true;
    var local$$65370 = new THREE.Vector3(local$$64732[_0x34b6[290]] + (local$$64739[_0x34b6[290]] - local$$64732[_0x34b6[290]]) / 2, local$$64732[_0x34b6[291]] + (local$$64739[_0x34b6[291]] - local$$64732[_0x34b6[291]]) / 2, local$$64732[_0x34b6[1287]] + (local$$64739[_0x34b6[1287]] - local$$64732[_0x34b6[1287]]) / 2);
    local$$65274[_0x34b6[430]][_0x34b6[338]](local$$65370);
    local$$65010 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$65274[_0x34b6[1912]]());
    local$$65031 = projectedRadius(1, getCamera()[_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$65010, getRenderer()[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    local$$65035 = 60 / local$$65031;
    local$$65274[_0x34b6[1090]][_0x34b6[334]](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[2030]][_0x34b6[220]](local$$65274);
    this[_0x34b6[1446]][_0x34b6[274]](local$$65274);
    this[_0x34b6[2026]] = new LSJGeoMarker;
    var local$$65450 = new LSJMarkerStyle;
    /** @type {number} */
    local$$65450[_0x34b6[1659]] = 25;
    var local$$65462 = local$$65450[_0x34b6[1672]]();
    local$$65462[_0x34b6[1652]](_0x34b6[2032]);
    local$$65462[_0x34b6[1406]]()[_0x34b6[1533]](1, 1, 1);
    local$$65462[_0x34b6[1656]](20);
    this[_0x34b6[2026]][_0x34b6[1670]](local$$65450);
    var local$$65497 = undefined;
    if (local$$64732[_0x34b6[1287]] < local$$64739[_0x34b6[1287]]) {
      local$$65497 = local$$64732[_0x34b6[212]]()[_0x34b6[1434]](local$$64743);
    } else {
      local$$65497 = local$$64739[_0x34b6[212]]()[_0x34b6[1434]](local$$64743);
    }
    this[_0x34b6[2026]][_0x34b6[1689]](local$$65497[_0x34b6[223]]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    this[_0x34b6[2026]][_0x34b6[1241]](local$$64743[_0x34b6[290]] + local$$65497[_0x34b6[290]] / 2, local$$64743[_0x34b6[291]] + local$$65497[_0x34b6[291]] / 2, local$$64743[_0x34b6[1287]] + local$$65497[_0x34b6[1287]] / 2);
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
    local$$65586[_0x34b6[2037]](local$$65497[_0x34b6[223]]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    /** @type {boolean} */
    local$$65586[_0x34b6[268]][_0x34b6[1307]] = false;
    /** @type {boolean} */
    local$$65586[_0x34b6[330]] = true;
    var local$$65669 = new THREE.Vector3(local$$64743[_0x34b6[290]] + local$$65497[_0x34b6[290]] / 2, local$$64743[_0x34b6[291]] + local$$65497[_0x34b6[291]] / 2, local$$64743[_0x34b6[1287]] + local$$65497[_0x34b6[1287]] / 2);
    local$$65586[_0x34b6[430]][_0x34b6[338]](local$$65669);
    local$$65010 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$65586[_0x34b6[1912]]());
    local$$65031 = projectedRadius(1, getCamera()[_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$65010, getRenderer()[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    local$$65035 = 60 / local$$65031;
    local$$65586[_0x34b6[1090]][_0x34b6[334]](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[2030]][_0x34b6[220]](local$$65586);
    this[_0x34b6[1446]][_0x34b6[274]](local$$65586);
    this[_0x34b6[2027]] = new LSJGeoMarker;
    var local$$65749 = new LSJMarkerStyle;
    /** @type {number} */
    local$$65749[_0x34b6[1659]] = 30;
    var local$$65761 = local$$65749[_0x34b6[1672]]();
    local$$65761[_0x34b6[1652]](_0x34b6[2032]);
    local$$65761[_0x34b6[1406]]()[_0x34b6[1533]](1, 1, 1);
    local$$65761[_0x34b6[1656]](20);
    this[_0x34b6[2027]][_0x34b6[1670]](local$$65749);
    if (local$$64732[_0x34b6[1287]] < local$$64739[_0x34b6[1287]]) {
      local$$65497 = local$$64739[_0x34b6[212]]()[_0x34b6[1434]](local$$64743);
    } else {
      local$$65497 = local$$64732[_0x34b6[212]]()[_0x34b6[1434]](local$$64743);
    }
    this[_0x34b6[2027]][_0x34b6[1689]](local$$65497[_0x34b6[223]]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    this[_0x34b6[2027]][_0x34b6[1241]](local$$64743[_0x34b6[290]] + local$$65497[_0x34b6[290]] / 2, local$$64743[_0x34b6[291]] + local$$65497[_0x34b6[291]] / 2, local$$64743[_0x34b6[1287]] + local$$65497[_0x34b6[1287]] / 2);
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
    local$$65883[_0x34b6[2037]](local$$65497[_0x34b6[223]]()[_0x34b6[2033]](2) + _0x34b6[2034]);
    /** @type {boolean} */
    local$$65883[_0x34b6[268]][_0x34b6[1307]] = false;
    /** @type {boolean} */
    local$$65883[_0x34b6[330]] = true;
    var local$$65966 = new THREE.Vector3(local$$64743[_0x34b6[290]] + local$$65497[_0x34b6[290]] / 2, local$$64743[_0x34b6[291]] + local$$65497[_0x34b6[291]] / 2, local$$64743[_0x34b6[1287]] + local$$65497[_0x34b6[1287]] / 2);
    local$$65883[_0x34b6[430]][_0x34b6[338]](local$$65966);
    local$$65010 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$65883[_0x34b6[1912]]());
    local$$65031 = projectedRadius(1, getCamera()[_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$65010, getRenderer()[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    local$$65035 = 60 / local$$65031;
    local$$65883[_0x34b6[1090]][_0x34b6[334]](local$$65035, local$$65035, local$$65035);
    this[_0x34b6[2030]][_0x34b6[220]](local$$65883);
    this[_0x34b6[1446]][_0x34b6[274]](local$$65883);
    local$$64690[_0x34b6[274]](this[_0x34b6[1446]]);
  }
  /** @type {boolean} */
  this[_0x34b6[2029]] = false;
};
/**
 * @param {?} local$$66066
 * @return {undefined}
 */
LSJRulerDistance[_0x34b6[219]][_0x34b6[225]] = function(local$$66066) {
  if (this[_0x34b6[2029]]) {
    this[_0x34b6[1261]](local$$66066[_0x34b6[2038]]());
  }
  if (this[_0x34b6[2028]] != undefined) {
    if (this[_0x34b6[1523]][_0x34b6[1126]][_0x34b6[1447]] === null) {
      this[_0x34b6[1523]][_0x34b6[1126]][_0x34b6[1716]]();
    }
    var local$$66124 = this[_0x34b6[1523]][_0x34b6[1126]][_0x34b6[1447]][_0x34b6[658]];
    var local$$66137 = local$$66124[_0x34b6[212]]()[_0x34b6[1477]](local$$66066[_0x34b6[1692]]);
    /** @type {number} */
    var local$$66156 = (local$$66137[_0x34b6[290]] + 1) / 2 * local$$66066[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[545]];
    /** @type {number} */
    var local$$66176 = -(local$$66137[_0x34b6[291]] - 1) / 2 * local$$66066[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
    /** @type {number} */
    local$$66137[_0x34b6[290]] = (local$$66156 + 1) / controlRender[_0x34b6[1694]][_0x34b6[545]] * 2 - 1;
    local$$66137[_0x34b6[1480]](local$$66066[_0x34b6[1692]]);
    var local$$66212 = local$$66137[_0x34b6[1434]](local$$66124)[_0x34b6[223]]();
    /** @type {number} */
    var local$$66227 = this[_0x34b6[1523]][_0x34b6[1126]][_0x34b6[1447]][_0x34b6[1693]] / local$$66212;
    /** @type {number} */
    var local$$66231 = local$$66227 / 6;
    /** @type {number} */
    var local$$66246 = this[_0x34b6[1523]][_0x34b6[1126]][_0x34b6[1447]][_0x34b6[1693]] / local$$66231;
    /** @type {number} */
    var local$$66248 = local$$66246;
    this[_0x34b6[2028]][_0x34b6[1529]]({
      dashSize : local$$66246,
      gapSize : local$$66248,
      linewidth : 3
    });
  }
  /** @type {number} */
  var local$$66264 = 0;
  for (; local$$66264 < this[_0x34b6[2030]][_0x34b6[223]]; local$$66264++) {
    var local$$66279 = this[_0x34b6[2030]][local$$66264];
    var local$$66295 = local$$66066[_0x34b6[1692]][_0x34b6[430]][_0x34b6[1593]](local$$66279[_0x34b6[1912]]());
    var local$$66317 = projectedRadius(1, local$$66066[_0x34b6[1692]][_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$66295, controlRender[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    var local$$66321 = 60 / local$$66317;
    local$$66279[_0x34b6[1090]][_0x34b6[334]](local$$66321, local$$66321, local$$66321);
  }
  /** @type {number} */
  var local$$66335 = 0;
  for (; local$$66335 < this[_0x34b6[1907]][_0x34b6[223]]; local$$66335++) {
    var local$$66350 = this[_0x34b6[1907]][local$$66335];
    local$$66295 = local$$66066[_0x34b6[1692]][_0x34b6[430]][_0x34b6[1593]](local$$66350[_0x34b6[1912]]());
    local$$66317 = projectedRadius(1, local$$66066[_0x34b6[1692]][_0x34b6[1913]] * Math[_0x34b6[979]] / 180, local$$66295, controlRender[_0x34b6[1694]][_0x34b6[548]]);
    /** @type {number} */
    local$$66321 = 10 / local$$66317;
    local$$66350[_0x34b6[1090]][_0x34b6[334]](local$$66321, local$$66321, local$$66321);
  }
};
(function() {
  /**
   * @param {?} local$$66412
   * @return {undefined}
   */
  function local$$66411(local$$66412) {
    this[_0x34b6[368]] = local$$66412;
    this[_0x34b6[1875]] = new THREE.Vector3;
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
    this[_0x34b6[1937]] = false;
    /** @type {number} */
    this[_0x34b6[1938]] = .005;
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
      var local$$66535 = (new THREE.Quaternion)[_0x34b6[2048]](this[_0x34b6[368]][_0x34b6[1936]], new THREE.Vector3(0, 1, 0));
      var local$$66545 = local$$66535[_0x34b6[212]]()[_0x34b6[241]]();
      var local$$66553 = this[_0x34b6[368]][_0x34b6[430]];
      var local$$66557 = new THREE.Vector3;
      local$$66557[_0x34b6[338]](local$$66553)[_0x34b6[1434]](this[_0x34b6[1875]]);
      local$$66557[_0x34b6[1696]](local$$66535);
      local$$66489 = Math[_0x34b6[1564]](local$$66557[_0x34b6[290]], local$$66557[_0x34b6[1287]]);
      local$$66492 = Math[_0x34b6[1564]](Math[_0x34b6[889]](local$$66557[_0x34b6[290]] * local$$66557[_0x34b6[290]] + local$$66557[_0x34b6[1287]] * local$$66557[_0x34b6[1287]]), local$$66557[_0x34b6[291]]);
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
        var local$$66682 = this[_0x34b6[368]][_0x34b6[740]][_0x34b6[1280]];
        local$$66668[_0x34b6[334]](local$$66682[0], local$$66682[1], local$$66682[2]);
        local$$66668[_0x34b6[350]](-local$$66671);
        local$$66505[_0x34b6[274]](local$$66668);
      };
    }();
    this[_0x34b6[2054]] = function() {
      var local$$66720 = new THREE.Vector3;
      return function local$$66722(local$$66723) {
        var local$$66734 = this[_0x34b6[368]][_0x34b6[740]][_0x34b6[1280]];
        local$$66720[_0x34b6[334]](local$$66734[4], local$$66734[5], local$$66734[6]);
        local$$66720[_0x34b6[350]](local$$66723);
        local$$66505[_0x34b6[274]](local$$66720);
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
      if (local$$66483[_0x34b6[368]] instanceof THREE[_0x34b6[2056]]) {
        var local$$66786 = local$$66483[_0x34b6[368]][_0x34b6[430]];
        var local$$66799 = local$$66786[_0x34b6[212]]()[_0x34b6[1434]](local$$66483[_0x34b6[1875]]);
        var local$$66805 = local$$66799[_0x34b6[223]]();
        /** @type {number} */
        local$$66805 = local$$66805 * Math[_0x34b6[1416]](local$$66483[_0x34b6[368]][_0x34b6[1913]] / 2 * Math[_0x34b6[979]] / 180);
        local$$66483[_0x34b6[2053]](2 * local$$66768 * local$$66805 / local$$66771);
        local$$66483[_0x34b6[2054]](2 * local$$66769 * local$$66805 / local$$66771);
      } else {
        if (local$$66483[_0x34b6[368]] instanceof THREE[_0x34b6[2057]]) {
          local$$66483[_0x34b6[2053]](local$$66768 * (local$$66483[_0x34b6[368]][_0x34b6[655]] - local$$66483[_0x34b6[368]][_0x34b6[432]]) / local$$66770);
          local$$66483[_0x34b6[2054]](local$$66769 * (local$$66483[_0x34b6[368]][_0x34b6[434]] - local$$66483[_0x34b6[368]][_0x34b6[656]]) / local$$66771);
        } else {
          console[_0x34b6[1063]](_0x34b6[2058]);
        }
      }
    };
    /**
     * @param {?} local$$66913
     * @return {undefined}
     */
    this[_0x34b6[2059]] = function(local$$66913) {
      if (local$$66483[_0x34b6[368]] instanceof THREE[_0x34b6[2056]]) {
        /** @type {number} */
        local$$66501 = local$$66501 / local$$66913;
      } else {
        if (local$$66483[_0x34b6[368]] instanceof THREE[_0x34b6[2057]]) {
          local$$66483[_0x34b6[368]][_0x34b6[2060]] = Math[_0x34b6[532]](this[_0x34b6[2041]], Math[_0x34b6[472]](this[_0x34b6[2042]], this[_0x34b6[368]][_0x34b6[2060]] * local$$66913));
          local$$66483[_0x34b6[368]][_0x34b6[1620]]();
          /** @type {boolean} */
          local$$66512 = true;
        } else {
          console[_0x34b6[1063]](_0x34b6[2061]);
        }
      }
    };
    /**
     * @param {?} local$$66992
     * @return {undefined}
     */
    this[_0x34b6[2062]] = function(local$$66992) {
      if (local$$66483[_0x34b6[368]] instanceof THREE[_0x34b6[2056]]) {
        /** @type {number} */
        local$$66501 = local$$66501 * local$$66992;
      } else {
        if (local$$66483[_0x34b6[368]] instanceof THREE[_0x34b6[2057]]) {
          local$$66483[_0x34b6[368]][_0x34b6[2060]] = Math[_0x34b6[532]](this[_0x34b6[2041]], Math[_0x34b6[472]](this[_0x34b6[2042]], this[_0x34b6[368]][_0x34b6[2060]] / local$$66992));
          local$$66483[_0x34b6[368]][_0x34b6[1620]]();
          /** @type {boolean} */
          local$$66512 = true;
        } else {
          console[_0x34b6[1063]](_0x34b6[2061]);
        }
      }
    };
    this[_0x34b6[1261]] = function() {
      var local$$67074 = new THREE.Vector3;
      var local$$67090 = (new THREE.Quaternion)[_0x34b6[2048]](local$$66412[_0x34b6[1936]], new THREE.Vector3(0, 1, 0));
      var local$$67100 = local$$67090[_0x34b6[212]]()[_0x34b6[241]]();
      var local$$67104 = new THREE.Vector3;
      var local$$67108 = new THREE.Quaternion;
      return function() {
        var local$$67117 = this[_0x34b6[368]][_0x34b6[430]];
        local$$67074[_0x34b6[338]](local$$67117)[_0x34b6[1434]](this[_0x34b6[1875]]);
        local$$67074[_0x34b6[1696]](local$$67090);
        local$$66489 = local$$66489 + local$$66498;
        local$$66492 = local$$66492 + local$$66495;
        local$$66489 = Math[_0x34b6[532]](this[_0x34b6[2045]], Math[_0x34b6[472]](this[_0x34b6[2046]], local$$66489));
        local$$66492 = Math[_0x34b6[532]](this[_0x34b6[2043]], Math[_0x34b6[472]](this[_0x34b6[2044]], local$$66492));
        /** @type {number} */
        var local$$67179 = local$$67074[_0x34b6[223]]() * local$$66501;
        local$$67179 = Math[_0x34b6[532]](this[_0x34b6[2039]], Math[_0x34b6[472]](this[_0x34b6[2040]], local$$67179));
        this[_0x34b6[1875]][_0x34b6[274]](local$$66505);
        /** @type {number} */
        local$$67074[_0x34b6[290]] = local$$67179 * Math[_0x34b6[1562]](local$$66492) * Math[_0x34b6[1562]](local$$66489);
        /** @type {number} */
        local$$67074[_0x34b6[291]] = local$$67179 * Math[_0x34b6[349]](local$$66492);
        /** @type {number} */
        local$$67074[_0x34b6[1287]] = local$$67179 * Math[_0x34b6[1562]](local$$66492) * Math[_0x34b6[349]](local$$66489);
        local$$67074[_0x34b6[1696]](local$$67100);
        local$$67117[_0x34b6[338]](this[_0x34b6[1875]])[_0x34b6[274]](local$$67074);
        var local$$67267 = new THREE.Vector3(0, 1, 0);
        var local$$67274 = new THREE.Vector3(1, 0, 0);
        local$$67267[_0x34b6[2063]](local$$67274, local$$66492);
        if (local$$67267[_0x34b6[1287]] < 0) {
          /** @type {number} */
          this[_0x34b6[368]][_0x34b6[1936]][_0x34b6[1287]] = -1;
          this[_0x34b6[368]][_0x34b6[1549]](this[_0x34b6[1875]]);
        } else {
          /** @type {number} */
          this[_0x34b6[368]][_0x34b6[1936]][_0x34b6[1287]] = 1;
          this[_0x34b6[368]][_0x34b6[1549]](this[_0x34b6[1875]]);
        }
        if (this[_0x34b6[1937]] === true) {
          /** @type {number} */
          local$$66498 = local$$66498 * (1 - this[_0x34b6[1938]]);
          /** @type {number} */
          local$$66495 = local$$66495 * (1 - this[_0x34b6[1938]]);
        } else {
          /** @type {number} */
          local$$66498 = 0;
          /** @type {number} */
          local$$66495 = 0;
        }
        /** @type {number} */
        local$$66501 = 1;
        local$$66505[_0x34b6[334]](0, 0, 0);
        if (local$$66512 || local$$67104[_0x34b6[2064]](this[_0x34b6[368]][_0x34b6[430]]) > local$$66486 || 8 * (1 - local$$67108[_0x34b6[1432]](this[_0x34b6[368]][_0x34b6[1239]])) > local$$66486) {
          local$$67104[_0x34b6[338]](this[_0x34b6[368]][_0x34b6[430]]);
          local$$67108[_0x34b6[338]](this[_0x34b6[368]][_0x34b6[1239]]);
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
      var local$$67473 = local$$67458[_0x34b6[1694]] === document ? local$$67458[_0x34b6[1694]][_0x34b6[440]] : local$$67458[_0x34b6[1694]];
      local$$67475[_0x34b6[2055]](local$$67455, local$$67456, local$$67473[_0x34b6[545]], local$$67473[_0x34b6[548]]);
    }
    /**
     * @return {?}
     */
    function local$$67489() {
      return 2 * Math[_0x34b6[979]] / 60 / 60 * local$$67458[_0x34b6[2074]];
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
      if (local$$67458[_0x34b6[224]] === false) {
        return;
      }
      local$$67521[_0x34b6[1428]]();
      if (local$$67521[_0x34b6[1594]] === local$$67458[_0x34b6[2077]][_0x34b6[2085]]) {
        if (local$$67458[_0x34b6[2069]] === false) {
          return;
        }
        local$$67556 = local$$67557[_0x34b6[2086]];
        local$$67563[_0x34b6[334]](local$$67521[_0x34b6[1429]], local$$67521[_0x34b6[1430]]);
      } else {
        if (local$$67521[_0x34b6[1594]] === local$$67458[_0x34b6[2077]][_0x34b6[2087]]) {
          if (local$$67458[_0x34b6[2071]] === false) {
            return;
          }
          local$$67556 = local$$67557[_0x34b6[2088]];
          local$$67600[_0x34b6[334]](local$$67521[_0x34b6[1429]], local$$67521[_0x34b6[1430]]);
        } else {
          if (local$$67521[_0x34b6[1594]] === local$$67458[_0x34b6[2077]][_0x34b6[2088]]) {
            if (local$$67458[_0x34b6[2071]] === false) {
              return;
            }
            local$$67556 = local$$67557[_0x34b6[2088]];
            local$$67600[_0x34b6[334]](local$$67521[_0x34b6[1429]], local$$67521[_0x34b6[1430]]);
          }
        }
      }
      if (local$$67556 !== local$$67557[_0x34b6[2079]]) {
        document[_0x34b6[1423]](_0x34b6[1422], local$$67664, false);
        document[_0x34b6[1423]](_0x34b6[1425], local$$67673, false);
        local$$67458[_0x34b6[1589]](local$$67680);
      }
    }
    /**
     * @param {?} local$$67687
     * @return {undefined}
     */
    function local$$67664(local$$67687) {
      if (local$$67458[_0x34b6[224]] === false) {
        return;
      }
      local$$67687[_0x34b6[1428]]();
      var local$$67717 = local$$67458[_0x34b6[1694]] === document ? local$$67458[_0x34b6[1694]][_0x34b6[440]] : local$$67458[_0x34b6[1694]];
      if (local$$67556 === local$$67557[_0x34b6[2086]]) {
        if (local$$67458[_0x34b6[2069]] === false) {
          return;
        }
        local$$67732[_0x34b6[334]](local$$67687[_0x34b6[1429]], local$$67687[_0x34b6[1430]]);
        local$$67744[_0x34b6[1697]](local$$67732, local$$67563);
        local$$67475[_0x34b6[2051]](2 * Math[_0x34b6[979]] * local$$67744[_0x34b6[290]] / local$$67717[_0x34b6[545]] * local$$67458[_0x34b6[2070]]);
        local$$67475[_0x34b6[2052]](2 * Math[_0x34b6[979]] * local$$67744[_0x34b6[291]] / local$$67717[_0x34b6[548]] * local$$67458[_0x34b6[2070]]);
        local$$67563[_0x34b6[338]](local$$67732);
      } else {
        if (local$$67556 === local$$67557[_0x34b6[2089]]) {
          if (local$$67458[_0x34b6[1939]] === false) {
            return;
          }
          local$$67813[_0x34b6[334]](local$$67687[_0x34b6[1429]], local$$67687[_0x34b6[1430]]);
          local$$67825[_0x34b6[1697]](local$$67813, local$$67829);
          if (local$$67825[_0x34b6[291]] > 0) {
            local$$67475[_0x34b6[2059]](local$$67507());
          } else {
            if (local$$67825[_0x34b6[291]] < 0) {
              local$$67475[_0x34b6[2062]](local$$67507());
            }
          }
          local$$67829[_0x34b6[338]](local$$67813);
        } else {
          if (local$$67556 === local$$67557[_0x34b6[2088]]) {
            if (local$$67458[_0x34b6[2071]] === false) {
              return;
            }
            local$$67879[_0x34b6[334]](local$$67687[_0x34b6[1429]], local$$67687[_0x34b6[1430]]);
            local$$67891[_0x34b6[1697]](local$$67879, local$$67600);
            local$$67454(local$$67891[_0x34b6[290]], local$$67891[_0x34b6[291]]);
            local$$67600[_0x34b6[338]](local$$67879);
          }
        }
      }
      if (local$$67556 !== local$$67557[_0x34b6[2079]]) {
        local$$67458[_0x34b6[1261]]();
      }
    }
    /**
     * @return {undefined}
     */
    function local$$67673() {
      if (local$$67458[_0x34b6[224]] === false) {
        return;
      }
      document[_0x34b6[1427]](_0x34b6[1422], local$$67664, false);
      document[_0x34b6[1427]](_0x34b6[1425], local$$67673, false);
      local$$67458[_0x34b6[1589]](local$$67959);
      local$$67556 = local$$67557[_0x34b6[2079]];
    }
    /**
     * @param {?} local$$67970
     * @return {undefined}
     */
    function local$$67969(local$$67970) {
      if (local$$67458[_0x34b6[224]] === false || local$$67458[_0x34b6[1939]] === false || local$$67556 !== local$$67557[_0x34b6[2079]]) {
        return;
      }
      local$$67970[_0x34b6[1428]]();
      local$$67970[_0x34b6[1596]]();
      /** @type {number} */
      var local$$68003 = 0;
      if (local$$67970[_0x34b6[1989]] !== undefined) {
        local$$68003 = local$$67970[_0x34b6[1989]];
      } else {
        if (local$$67970[_0x34b6[1990]] !== undefined) {
          /** @type {number} */
          local$$68003 = -local$$67970[_0x34b6[1990]];
        }
      }
      if (local$$68003 > 0) {
        local$$67475[_0x34b6[2062]](local$$67507());
      } else {
        if (local$$68003 < 0) {
          local$$67475[_0x34b6[2059]](local$$67507());
        }
      }
      local$$67458[_0x34b6[1261]]();
      local$$67458[_0x34b6[1589]](local$$67680);
      local$$67458[_0x34b6[1589]](local$$67959);
    }
    /**
     * @param {?} local$$68070
     * @return {undefined}
     */
    function local$$68069(local$$68070) {
      if (local$$67458[_0x34b6[224]] === false || local$$67458[_0x34b6[2075]] === false || local$$67458[_0x34b6[2071]] === false) {
        return;
      }
      switch(local$$68070[_0x34b6[2092]]) {
        case local$$67458[_0x34b6[2076]][_0x34b6[2090]]:
          local$$67454(0, local$$67458[_0x34b6[2072]]);
          local$$67458[_0x34b6[1261]]();
          break;
        case local$$67458[_0x34b6[2076]][_0x34b6[2091]]:
          local$$67454(0, -local$$67458[_0x34b6[2072]]);
          local$$67458[_0x34b6[1261]]();
          break;
        case local$$67458[_0x34b6[2076]][_0x34b6[1954]]:
          local$$67454(local$$67458[_0x34b6[2072]], 0);
          local$$67458[_0x34b6[1261]]();
          break;
        case local$$67458[_0x34b6[2076]][_0x34b6[1956]]:
          local$$67454(-local$$67458[_0x34b6[2072]], 0);
          local$$67458[_0x34b6[1261]]();
          break;
      }
    }
    /**
     * @param {?} local$$68182
     * @return {undefined}
     */
    function local$$68181(local$$68182) {
      if (local$$67458[_0x34b6[224]] === false) {
        return;
      }
      switch(local$$68182[_0x34b6[1984]][_0x34b6[223]]) {
        case 1:
          if (local$$67458[_0x34b6[2069]] === false) {
            return;
          }
          local$$67556 = local$$67557[_0x34b6[2093]];
          local$$67563[_0x34b6[334]](local$$68182[_0x34b6[1984]][0][_0x34b6[2094]], local$$68182[_0x34b6[1984]][0][_0x34b6[2095]]);
          break;
        case 2:
          if (local$$67458[_0x34b6[1939]] === false) {
            return;
          }
          /** @type {number} */
          var local$$68265 = local$$68182[_0x34b6[1984]][0][_0x34b6[2094]] - local$$68182[_0x34b6[1984]][1][_0x34b6[2094]];
          /** @type {number} */
          var local$$68284 = local$$68182[_0x34b6[1984]][0][_0x34b6[2095]] - local$$68182[_0x34b6[1984]][1][_0x34b6[2095]];
          var local$$68293 = Math[_0x34b6[889]](local$$68265 * local$$68265 + local$$68284 * local$$68284);
          local$$67829[_0x34b6[334]](0, local$$68293);
          local$$68301[_0x34b6[334]](local$$68182[_0x34b6[1984]][0][_0x34b6[2094]], local$$68182[_0x34b6[1984]][0][_0x34b6[2095]]);
          local$$68323[_0x34b6[334]](local$$68182[_0x34b6[1984]][1][_0x34b6[2094]], local$$68182[_0x34b6[1984]][1][_0x34b6[2095]]);
          if (local$$67458[_0x34b6[2071]] === false) {
            return;
          }
          local$$67600[_0x34b6[334]](local$$68182[_0x34b6[1984]][0][_0x34b6[2094]], local$$68182[_0x34b6[1984]][0][_0x34b6[2095]]);
          break;
        default:
          local$$67556 = local$$67557[_0x34b6[2079]];
      }
      if (local$$67556 !== local$$67557[_0x34b6[2079]]) {
        local$$67458[_0x34b6[1589]](local$$67680);
      }
    }
    /**
     * @param {?} local$$68401
     * @return {undefined}
     */
    function local$$68400(local$$68401) {
      if (local$$67458[_0x34b6[224]] === false) {
        return;
      }
      local$$68401[_0x34b6[1428]]();
      local$$68401[_0x34b6[1596]]();
      var local$$68436 = local$$67458[_0x34b6[1694]] === document ? local$$67458[_0x34b6[1694]][_0x34b6[440]] : local$$67458[_0x34b6[1694]];
      switch(local$$68401[_0x34b6[1984]][_0x34b6[223]]) {
        case 1:
          if (local$$67458[_0x34b6[2069]] === false) {
            return;
          }
          if (local$$67556 !== local$$67557[_0x34b6[2093]]) {
            return;
          }
          local$$67732[_0x34b6[334]](local$$68401[_0x34b6[1984]][0][_0x34b6[2094]], local$$68401[_0x34b6[1984]][0][_0x34b6[2095]]);
          local$$67744[_0x34b6[1697]](local$$67732, local$$67563);
          local$$67475[_0x34b6[2051]](2 * Math[_0x34b6[979]] * local$$67744[_0x34b6[290]] / local$$68436[_0x34b6[545]] * local$$67458[_0x34b6[2070]]);
          local$$67475[_0x34b6[2052]](2 * Math[_0x34b6[979]] * local$$67744[_0x34b6[291]] / local$$68436[_0x34b6[548]] * local$$67458[_0x34b6[2070]]);
          local$$67563[_0x34b6[338]](local$$67732);
          local$$67458[_0x34b6[1261]]();
          break;
        case 2:
          if (local$$67458[_0x34b6[1939]] === false) {
            return;
          }
          /** @type {number} */
          var local$$68572 = local$$68401[_0x34b6[1984]][0][_0x34b6[2094]] - local$$68401[_0x34b6[1984]][1][_0x34b6[2094]];
          /** @type {number} */
          var local$$68591 = local$$68401[_0x34b6[1984]][0][_0x34b6[2095]] - local$$68401[_0x34b6[1984]][1][_0x34b6[2095]];
          var local$$68600 = Math[_0x34b6[889]](local$$68572 * local$$68572 + local$$68591 * local$$68591);
          var local$$68604 = new THREE.Vector2;
          var local$$68608 = new THREE.Vector2;
          local$$68604[_0x34b6[334]](local$$68401[_0x34b6[1984]][0][_0x34b6[2094]], local$$68401[_0x34b6[1984]][0][_0x34b6[2095]]);
          local$$68608[_0x34b6[334]](local$$68401[_0x34b6[1984]][1][_0x34b6[2094]], local$$68401[_0x34b6[1984]][1][_0x34b6[2095]]);
          local$$68604[_0x34b6[1697]](local$$68604, local$$68301);
          local$$68608[_0x34b6[1697]](local$$68608, local$$68323);
          local$$68301[_0x34b6[334]](local$$68401[_0x34b6[1984]][0][_0x34b6[2094]], local$$68401[_0x34b6[1984]][0][_0x34b6[2095]]);
          local$$68323[_0x34b6[334]](local$$68401[_0x34b6[1984]][1][_0x34b6[2094]], local$$68401[_0x34b6[1984]][1][_0x34b6[2095]]);
          local$$67813[_0x34b6[334]](0, local$$68600);
          local$$67825[_0x34b6[1697]](local$$67813, local$$67829);
          local$$67879[_0x34b6[334]](local$$68401[_0x34b6[1984]][0][_0x34b6[2094]], local$$68401[_0x34b6[1984]][0][_0x34b6[2095]]);
          local$$67829[_0x34b6[338]](local$$67813);
          if (Math[_0x34b6[1525]](local$$67825[_0x34b6[291]]) > 2) {
            local$$67556 = local$$67557[_0x34b6[2096]];
            if (local$$67825[_0x34b6[291]] > 0) {
              local$$67475[_0x34b6[2062]](local$$67507());
            } else {
              if (local$$67825[_0x34b6[291]] < 0) {
                local$$67475[_0x34b6[2059]](local$$67507());
              }
            }
          } else {
            if (local$$68604[_0x34b6[223]]() > 2 && local$$68608[_0x34b6[223]]() > 2) {
              if (local$$67458[_0x34b6[2071]] === false) {
                return;
              }
              local$$67556 = local$$67557[_0x34b6[2097]];
              local$$67891[_0x34b6[1697]](local$$67879, local$$67600);
              local$$67454(local$$67891[_0x34b6[290]], local$$67891[_0x34b6[291]]);
            }
          }
          local$$67600[_0x34b6[338]](local$$67879);
          local$$67458[_0x34b6[1261]]();
          break;
        default:
          local$$67556 = local$$67557[_0x34b6[2079]];
      }
    }
    /**
     * @return {undefined}
     */
    function local$$68851() {
      if (local$$67458[_0x34b6[224]] === false) {
        return;
      }
      local$$67458[_0x34b6[1589]](local$$67959);
      local$$67556 = local$$67557[_0x34b6[2079]];
    }
    /**
     * @param {?} local$$68875
     * @return {undefined}
     */
    function local$$68874(local$$68875) {
      local$$68875[_0x34b6[1428]]();
    }
    var local$$67475 = new local$$66411(local$$67451);
    this[_0x34b6[1694]] = local$$67452 !== undefined ? local$$67452 : document;
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
    this[_0x34b6[224]] = true;
    this[_0x34b6[658]] = this[_0x34b6[1875]];
    /** @type {boolean} */
    this[_0x34b6[1939]] = true;
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
      ORBIT : THREE[_0x34b6[1955]][_0x34b6[1954]],
      ZOOM : THREE[_0x34b6[1955]][_0x34b6[2078]],
      PAN : THREE[_0x34b6[1955]][_0x34b6[1956]]
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
    this[_0x34b6[2080]] = this[_0x34b6[1875]][_0x34b6[212]]();
    this[_0x34b6[2081]] = this[_0x34b6[368]][_0x34b6[430]][_0x34b6[212]]();
    this[_0x34b6[2082]] = this[_0x34b6[368]][_0x34b6[2060]];
    var local$$69121 = {
      type : _0x34b6[1575]
    };
    var local$$67680 = {
      type : _0x34b6[495]
    };
    var local$$67959 = {
      type : _0x34b6[899]
    };
    /**
     * @return {undefined}
     */
    this[_0x34b6[1261]] = function() {
      if (this[_0x34b6[2073]] && local$$67556 === local$$67557[_0x34b6[2079]]) {
        local$$67475[_0x34b6[2051]](local$$67489());
      }
      if (local$$67475[_0x34b6[1261]]() === true) {
        this[_0x34b6[1589]](local$$69121);
      }
    };
    /**
     * @return {undefined}
     */
    this[_0x34b6[2083]] = function() {
      local$$67556 = local$$67557[_0x34b6[2079]];
      this[_0x34b6[1875]][_0x34b6[338]](this[_0x34b6[2080]]);
      this[_0x34b6[368]][_0x34b6[430]][_0x34b6[338]](this[_0x34b6[2081]]);
      this[_0x34b6[368]][_0x34b6[2060]] = this[_0x34b6[2082]];
      this[_0x34b6[368]][_0x34b6[1620]]();
      this[_0x34b6[1589]](local$$69121);
      this[_0x34b6[1261]]();
    };
    /**
     * @return {undefined}
     */
    this[_0x34b6[232]] = function() {
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1992], local$$68874, false);
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1424], local$$67520, false);
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1991], local$$67969, false);
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1993], local$$67969, false);
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1579], local$$68181, false);
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1582], local$$68851, false);
      this[_0x34b6[1694]][_0x34b6[1427]](_0x34b6[1580], local$$68400, false);
      document[_0x34b6[1427]](_0x34b6[1422], local$$67664, false);
      document[_0x34b6[1427]](_0x34b6[1425], local$$67673, false);
      window[_0x34b6[1427]](_0x34b6[2098], local$$68069, false);
    };
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1992], local$$68874, false);
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1424], local$$67520, false);
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1991], local$$67969, false);
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1993], local$$67969, false);
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1579], local$$68181, false);
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1582], local$$68851, false);
    this[_0x34b6[1694]][_0x34b6[1423]](_0x34b6[1580], local$$68400, false);
    window[_0x34b6[1423]](_0x34b6[2098], local$$68069, false);
    this[_0x34b6[1261]]();
  };
  THREE[_0x34b6[2065]][_0x34b6[219]] = Object[_0x34b6[242]](THREE[_0x34b6[1057]][_0x34b6[219]]);
  THREE[_0x34b6[2065]][_0x34b6[219]][_0x34b6[1183]] = THREE[_0x34b6[2065]];
  Object[_0x34b6[2106]](THREE[_0x34b6[2065]][_0x34b6[219]], {
    object : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[368]];
      }
    },
    target : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[1875]];
      },
      set : function(local$$69501) {
        console[_0x34b6[1063]](_0x34b6[2099]);
        this[_0x34b6[2066]][_0x34b6[1875]][_0x34b6[338]](local$$69501);
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
        return this[_0x34b6[2066]][_0x34b6[1937]];
      },
      set : function(local$$69726) {
        this[_0x34b6[2066]][_0x34b6[1937]] = local$$69726;
      }
    },
    dampingFactor : {
      get : function() {
        return this[_0x34b6[2066]][_0x34b6[1938]];
      },
      set : function(local$$69749) {
        this[_0x34b6[2066]][_0x34b6[1938]] = local$$69749;
      }
    },
    noZoom : {
      get : function() {
        console[_0x34b6[1063]](_0x34b6[2100]);
        return !this[_0x34b6[1939]];
      },
      set : function(local$$69777) {
        console[_0x34b6[1063]](_0x34b6[2100]);
        /** @type {boolean} */
        this[_0x34b6[1939]] = !local$$69777;
      }
    },
    noRotate : {
      get : function() {
        console[_0x34b6[1063]](_0x34b6[2101]);
        return !this[_0x34b6[2069]];
      },
      set : function(local$$69811) {
        console[_0x34b6[1063]](_0x34b6[2101]);
        /** @type {boolean} */
        this[_0x34b6[2069]] = !local$$69811;
      }
    },
    noPan : {
      get : function() {
        console[_0x34b6[1063]](_0x34b6[2102]);
        return !this[_0x34b6[2071]];
      },
      set : function(local$$69845) {
        console[_0x34b6[1063]](_0x34b6[2102]);
        /** @type {boolean} */
        this[_0x34b6[2071]] = !local$$69845;
      }
    },
    noKeys : {
      get : function() {
        console[_0x34b6[1063]](_0x34b6[2103]);
        return !this[_0x34b6[2075]];
      },
      set : function(local$$69879) {
        console[_0x34b6[1063]](_0x34b6[2103]);
        /** @type {boolean} */
        this[_0x34b6[2075]] = !local$$69879;
      }
    },
    staticMoving : {
      get : function() {
        console[_0x34b6[1063]](_0x34b6[2104]);
        return !this[_0x34b6[2066]][_0x34b6[1937]];
      },
      set : function(local$$69916) {
        console[_0x34b6[1063]](_0x34b6[2104]);
        /** @type {boolean} */
        this[_0x34b6[2066]][_0x34b6[1937]] = !local$$69916;
      }
    },
    dynamicDampingFactor : {
      get : function() {
        console[_0x34b6[1063]](_0x34b6[2105]);
        return this[_0x34b6[2066]][_0x34b6[1938]];
      },
      set : function(local$$69955) {
        console[_0x34b6[1063]](_0x34b6[2105]);
        this[_0x34b6[2066]][_0x34b6[1938]] = local$$69955;
      }
    }
  });
})();
/**
 * @param {?} local$$69985
 * @return {undefined}
 */
LSJTextSprite = function(local$$69985) {
  THREE[_0x34b6[1348]][_0x34b6[238]](this);
  var local$$69995 = this;
  var local$$69999 = new THREE.Texture;
  local$$69999[_0x34b6[1820]] = THREE[_0x34b6[205]];
  local$$69999[_0x34b6[1821]] = THREE[_0x34b6[205]];
  var local$$70024 = new THREE.SpriteMaterial({
    map : local$$69999,
    useScreenCoordinates : true,
    depthTest : false,
    depthWrite : false
  });
  this[_0x34b6[268]] = local$$70024;
  this[_0x34b6[2107]] = new THREE.Sprite(local$$70024);
  this[_0x34b6[274]](this[_0x34b6[2107]]);
  /** @type {number} */
  this[_0x34b6[2108]] = 4;
  this[_0x34b6[2109]] = _0x34b6[2032];
  /** @type {number} */
  this[_0x34b6[1411]] = 28;
  this[_0x34b6[2110]] = {
    r : 0,
    g : 0,
    b : 0,
    a : 1
  };
  this[_0x34b6[751]] = {
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
  this[_0x34b6[739]] = _0x34b6[381];
  this[_0x34b6[2037]](local$$69985);
};
LSJTextSprite[_0x34b6[219]] = new THREE.Object3D;
/**
 * @param {?} local$$70124
 * @return {undefined}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[2037]] = function(local$$70124) {
  if (this[_0x34b6[739]] !== local$$70124) {
    this[_0x34b6[739]] = local$$70124;
    this[_0x34b6[1261]]();
  }
};
/**
 * @return {?}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[2112]] = function() {
  return this[_0x34b6[2107]];
};
/**
 * @param {?} local$$70167
 * @return {undefined}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[2113]] = function(local$$70167) {
  this[_0x34b6[2111]] = local$$70167;
  this[_0x34b6[1261]]();
};
/**
 * @param {?} local$$70189
 * @return {undefined}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[2035]] = function(local$$70189) {
  this[_0x34b6[2110]] = local$$70189;
  this[_0x34b6[1261]]();
};
/**
 * @param {?} local$$70211
 * @return {undefined}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[2036]] = function(local$$70211) {
  this[_0x34b6[751]] = local$$70211;
  this[_0x34b6[1261]]();
};
/**
 * @return {undefined}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[1261]] = function() {
  var local$$70240 = document[_0x34b6[424]](_0x34b6[516]);
  var local$$70248 = local$$70240[_0x34b6[403]](_0x34b6[402]);
  local$$70248[_0x34b6[834]] = _0x34b6[1392] + this[_0x34b6[1411]] + _0x34b6[1394] + this[_0x34b6[2109]];
  var local$$70275 = local$$70248[_0x34b6[1408]](this[_0x34b6[739]]);
  var local$$70280 = local$$70275[_0x34b6[208]];
  /** @type {number} */
  var local$$70283 = 5;
  var local$$70294 = 2 * local$$70283 + local$$70280 + 2 * this[_0x34b6[2108]];
  /** @type {number} */
  var local$$70307 = this[_0x34b6[1411]] * 1.4 + 2 * this[_0x34b6[2108]];
  local$$70240 = document[_0x34b6[424]](_0x34b6[516]);
  local$$70248 = local$$70240[_0x34b6[403]](_0x34b6[402]);
  local$$70248[_0x34b6[516]][_0x34b6[208]] = local$$70294;
  /** @type {number} */
  local$$70248[_0x34b6[516]][_0x34b6[209]] = local$$70307;
  local$$70248[_0x34b6[834]] = _0x34b6[1392] + this[_0x34b6[1411]] + _0x34b6[1394] + this[_0x34b6[2109]];
  local$$70248[_0x34b6[976]] = _0x34b6[476] + this[_0x34b6[751]][_0x34b6[458]] + _0x34b6[477] + this[_0x34b6[751]][_0x34b6[459]] + _0x34b6[477] + this[_0x34b6[751]][_0x34b6[460]] + _0x34b6[477] + this[_0x34b6[751]][_0x34b6[461]] + _0x34b6[478];
  local$$70248[_0x34b6[983]] = _0x34b6[476] + this[_0x34b6[2110]][_0x34b6[458]] + _0x34b6[477] + this[_0x34b6[2110]][_0x34b6[459]] + _0x34b6[477] + this[_0x34b6[2110]][_0x34b6[460]] + _0x34b6[477] + this[_0x34b6[2110]][_0x34b6[461]] + _0x34b6[478];
  local$$70248[_0x34b6[572]] = this[_0x34b6[2108]];
  this[_0x34b6[2114]](local$$70248, this[_0x34b6[2108]] / 2, this[_0x34b6[2108]] / 2, local$$70280 + this[_0x34b6[2108]] + 2 * local$$70283, this[_0x34b6[1411]] * 1.4 + this[_0x34b6[2108]], 6);
  local$$70248[_0x34b6[983]] = _0x34b6[2115];
  local$$70248[_0x34b6[1405]](this[_0x34b6[739]], this[_0x34b6[2108]] + local$$70283, this[_0x34b6[1411]] + this[_0x34b6[2108]]);
  local$$70248[_0x34b6[976]] = _0x34b6[476] + this[_0x34b6[2111]][_0x34b6[458]] + _0x34b6[477] + this[_0x34b6[2111]][_0x34b6[459]] + _0x34b6[477] + this[_0x34b6[2111]][_0x34b6[460]] + _0x34b6[477] + this[_0x34b6[2111]][_0x34b6[461]] + _0x34b6[478];
  local$$70248[_0x34b6[997]](this[_0x34b6[739]], this[_0x34b6[2108]] + local$$70283, this[_0x34b6[1411]] + this[_0x34b6[2108]]);
  var local$$70587 = new THREE.Texture(local$$70240);
  local$$70587[_0x34b6[1820]] = THREE[_0x34b6[205]];
  local$$70587[_0x34b6[1821]] = THREE[_0x34b6[205]];
  /** @type {boolean} */
  local$$70587[_0x34b6[1275]] = true;
  this[_0x34b6[2107]][_0x34b6[268]][_0x34b6[645]] = local$$70587;
  this[_0x34b6[2107]][_0x34b6[1090]][_0x34b6[334]](local$$70294 * .01, local$$70307 * .01, 1);
};
/**
 * @return {?}
 */
LSJTextSprite[_0x34b6[219]][_0x34b6[645]] = function() {
  return this[_0x34b6[2107]][_0x34b6[268]][_0x34b6[645]];
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
LSJTextSprite[_0x34b6[219]][_0x34b6[2114]] = function(local$$70669, local$$70670, local$$70671, local$$70672, local$$70673, local$$70674) {
  local$$70669[_0x34b6[978]]();
  local$$70669[_0x34b6[987]](local$$70670 + local$$70674, local$$70671);
  local$$70669[_0x34b6[2116]](local$$70670 + local$$70672 - local$$70674, local$$70671);
  local$$70669[_0x34b6[2117]](local$$70670 + local$$70672, local$$70671, local$$70670 + local$$70672, local$$70671 + local$$70674);
  local$$70669[_0x34b6[2116]](local$$70670 + local$$70672, local$$70671 + local$$70673 - local$$70674);
  local$$70669[_0x34b6[2117]](local$$70670 + local$$70672, local$$70671 + local$$70673, local$$70670 + local$$70672 - local$$70674, local$$70671 + local$$70673);
  local$$70669[_0x34b6[2116]](local$$70670 + local$$70674, local$$70671 + local$$70673);
  local$$70669[_0x34b6[2117]](local$$70670, local$$70671 + local$$70673, local$$70670, local$$70671 + local$$70673 - local$$70674);
  local$$70669[_0x34b6[2116]](local$$70670, local$$70671 + local$$70674);
  local$$70669[_0x34b6[2117]](local$$70670, local$$70671, local$$70670 + local$$70674, local$$70671);
  local$$70669[_0x34b6[981]]();
  local$$70669[_0x34b6[982]]();
  local$$70669[_0x34b6[984]]();
};
/**
 * @return {undefined}
 */
LSJModelNodeTexture = function() {
  /** @type {number} */
  this[_0x34b6[332]] = -1;
  this[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1741]];
  this[_0x34b6[1800]] = _0x34b6[381];
  /** @type {null} */
  this[_0x34b6[276]] = null;
  /** @type {boolean} */
  this[_0x34b6[1801]] = false;
  /** @type {boolean} */
  this[_0x34b6[2118]] = false;
};
/**
 * @return {undefined}
 */
LSJModelNodeMaterial = function() {
  /** @type {null} */
  this[_0x34b6[268]] = null;
  /** @type {!Array} */
  this[_0x34b6[2119]] = [];
};
/**
 * @return {undefined}
 */
LSJModelLODNode = function() {
  this[_0x34b6[445]] = _0x34b6[2120];
  /** @type {!Array} */
  this[_0x34b6[684]] = [];
  /** @type {!Array} */
  this[_0x34b6[1803]] = [];
  /** @type {null} */
  this[_0x34b6[2121]] = null;
  /** @type {null} */
  this[_0x34b6[667]] = null;
  /** @type {null} */
  this[_0x34b6[1762]] = null;
  this[_0x34b6[1785]] = _0x34b6[381];
  this[_0x34b6[1446]] = new THREE.Group;
  /** @type {boolean} */
  this[_0x34b6[1446]][_0x34b6[1714]] = true;
  /** @type {boolean} */
  this[_0x34b6[1446]][_0x34b6[1715]] = true;
  /** @type {boolean} */
  this[_0x34b6[1804]] = false;
  /** @type {boolean} */
  this[_0x34b6[1805]] = false;
  this[_0x34b6[1759]] = new THREE.Sphere;
  this[_0x34b6[1806]] = new THREE.Box3;
  this[_0x34b6[1807]] = LSELoadStatus[_0x34b6[1741]];
  /** @type {number} */
  this[_0x34b6[1808]] = 0;
  /** @type {number} */
  this[_0x34b6[1750]] = 0;
  /** @type {number} */
  this[_0x34b6[1751]] = 0;
  /** @type {boolean} */
  this[_0x34b6[1809]] = false;
  /** @type {!Array} */
  this[_0x34b6[1810]] = [];
  /** @type {!Array} */
  this[_0x34b6[2122]] = [];
  /** @type {!Array} */
  this[_0x34b6[1811]] = [];
  /** @type {!Array} */
  this[_0x34b6[2123]] = [];
  /** @type {null} */
  this[_0x34b6[1812]] = null;
  /** @type {number} */
  this[_0x34b6[1799]] = 0;
  /** @type {boolean} */
  this[_0x34b6[2124]] = false;
  /** @type {boolean} */
  this[_0x34b6[2125]] = false;
};
/** @type {function(): undefined} */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1183]] = LSJModelLODNode;
/**
 * @param {?} local$$71026
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1814]] = function(local$$71026) {
  this[_0x34b6[1805]] = local$$71026;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1815]] = function() {
  return this[_0x34b6[1805]];
};
/**
 * @param {?} local$$71058
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1816]] = function(local$$71058) {
  this[_0x34b6[1807]] = local$$71058;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1817]] = function() {
  return this[_0x34b6[1809]];
};
/**
 * @param {?} local$$71090
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1818]] = function(local$$71090) {
  this[_0x34b6[1809]] = local$$71090;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1792]] = function() {
  return this[_0x34b6[1807]];
};
/**
 * @param {?} local$$71122
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1769]] = function(local$$71122) {
  this[_0x34b6[1751]] = local$$71122;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1770]] = function() {
  return this[_0x34b6[1751]];
};
/**
 * @param {?} local$$71154
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1771]] = function(local$$71154) {
  this[_0x34b6[1750]] = local$$71154;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1772]] = function() {
  return this[_0x34b6[1750]];
};
/**
 * @param {?} local$$71186
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[2126]] = function(local$$71186) {
  this[_0x34b6[2125]] = local$$71186;
  var local$$71199 = this[_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  var local$$71202 = 0;
  for (; local$$71202 < local$$71199; local$$71202++) {
    var local$$71211 = this[_0x34b6[684]][local$$71202];
    local$$71211[_0x34b6[2126]](local$$71186);
  }
};
/**
 * @param {?} local$$71230
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1760]] = function(local$$71230) {
  this[_0x34b6[684]][_0x34b6[220]](local$$71230);
  local$$71230[_0x34b6[2121]] = this[_0x34b6[2121]];
  local$$71230[_0x34b6[1762]] = this[_0x34b6[1762]];
  local$$71230[_0x34b6[667]] = this;
  this[_0x34b6[1446]][_0x34b6[274]](local$$71230[_0x34b6[1446]]);
};
/**
 * @param {?} local$$71282
 * @param {?} local$$71283
 * @param {?} local$$71284
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1089]] = function(local$$71282, local$$71283, local$$71284) {
  if (local$$71283[_0x34b6[1755]] > local$$71283[_0x34b6[1754]]) {
    return;
  }
  var local$$71300 = local$$71282[_0x34b6[276]];
  if (local$$71300 == null || local$$71300 === undefined) {
    local$$71282[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    return null;
  }
  local$$71282[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1775]];
  local$$71283[_0x34b6[1755]]++;
  var local$$71335 = local$$71282[_0x34b6[1800]];
  var local$$71354 = local$$71335[_0x34b6[474]](local$$71335[_0x34b6[382]](_0x34b6[378]), local$$71335[_0x34b6[223]])[_0x34b6[387]]();
  if (local$$71354 == _0x34b6[2127]) {
    local$$71282[_0x34b6[276]] = this[_0x34b6[2128]](local$$71282, local$$71283, local$$71284);
  } else {
    if (local$$71354 == _0x34b6[2129]) {
      local$$71282[_0x34b6[276]] = this[_0x34b6[2130]](local$$71282, local$$71283, local$$71284);
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
        if (local$$71282[_0x34b6[1801]]) {
          window[_0x34b6[579]][_0x34b6[1819]](local$$71282[_0x34b6[1800]]);
        }
        local$$71282[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
        local$$71283[_0x34b6[1755]]--;
      };
      var local$$71437 = THREE[_0x34b6[1110]][_0x34b6[1109]][_0x34b6[640]](local$$71282[_0x34b6[1800]]);
      var local$$71442 = THREE[_0x34b6[1056]];
      if (local$$71437 !== null) {
        local$$71300 = local$$71437[_0x34b6[1060]](local$$71282[_0x34b6[1800]], local$$71284);
      } else {
        local$$71437 = new THREE.ImageLoader(local$$71442);
        local$$71437[_0x34b6[1072]](_0x34b6[381]);
        local$$71437[_0x34b6[1060]](local$$71282[_0x34b6[1800]], function(local$$71474) {
          local$$71300[_0x34b6[554]] = local$$71474;
          /** @type {boolean} */
          local$$71300[_0x34b6[1275]] = true;
          local$$71300[_0x34b6[294]] = THREE[_0x34b6[1081]];
          local$$71300[_0x34b6[1092]] = THREE[_0x34b6[1083]];
          local$$71300[_0x34b6[1093]] = THREE[_0x34b6[1083]];
          local$$71300[_0x34b6[1820]] = THREE[_0x34b6[2131]];
          local$$71300[_0x34b6[1821]] = THREE[_0x34b6[205]];
          /** @type {boolean} */
          local$$71300[_0x34b6[297]] = true;
          if (local$$71282[_0x34b6[1801]]) {
            window[_0x34b6[579]][_0x34b6[1819]](local$$71282[_0x34b6[1800]]);
          }
          local$$71282[_0x34b6[1800]] = _0x34b6[381];
          local$$71282[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
          local$$71283[_0x34b6[1755]]--;
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
LSJModelLODNode[_0x34b6[219]][_0x34b6[2128]] = function(local$$71592, local$$71593, local$$71594) {
  /**
   * @return {undefined}
   */
  var local$$71599 = function() {
  };
  /**
   * @return {undefined}
   */
  var local$$71634 = function() {
    if (local$$71592[_0x34b6[1801]]) {
      window[_0x34b6[579]][_0x34b6[1819]](local$$71592[_0x34b6[1800]]);
    }
    local$$71592[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    local$$71593[_0x34b6[1755]]--;
  };
  /**
   * @param {?} local$$71636
   * @return {undefined}
   */
  local$$71594 = function(local$$71636) {
    /** @type {boolean} */
    local$$71636[_0x34b6[1275]] = true;
    local$$71636[_0x34b6[294]] = THREE[_0x34b6[1081]];
    local$$71636[_0x34b6[1092]] = THREE[_0x34b6[1083]];
    local$$71636[_0x34b6[1093]] = THREE[_0x34b6[1083]];
    local$$71636[_0x34b6[1820]] = THREE[_0x34b6[205]];
    local$$71636[_0x34b6[1821]] = THREE[_0x34b6[205]];
    /** @type {boolean} */
    local$$71636[_0x34b6[297]] = false;
    if (local$$71592[_0x34b6[1801]]) {
      window[_0x34b6[579]][_0x34b6[1819]](local$$71592[_0x34b6[1800]]);
    }
    local$$71592[_0x34b6[1800]] = _0x34b6[381];
    local$$71592[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    local$$71593[_0x34b6[1755]]--;
  };
  var local$$71733 = new THREE.PVRLoader;
  var local$$71742 = local$$71733[_0x34b6[1060]](local$$71592[_0x34b6[1800]], local$$71594, local$$71599, local$$71634);
  return local$$71742;
};
/**
 * @param {?} local$$71755
 * @param {?} local$$71756
 * @param {!Function} local$$71757
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[2130]] = function(local$$71755, local$$71756, local$$71757) {
  /**
   * @return {undefined}
   */
  var local$$71762 = function() {
  };
  /**
   * @return {undefined}
   */
  var local$$71797 = function() {
    if (local$$71755[_0x34b6[1801]]) {
      window[_0x34b6[579]][_0x34b6[1819]](local$$71755[_0x34b6[1800]]);
    }
    local$$71755[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    local$$71756[_0x34b6[1755]]--;
  };
  /**
   * @param {?} local$$71799
   * @return {undefined}
   */
  local$$71757 = function(local$$71799) {
    /** @type {boolean} */
    local$$71799[_0x34b6[1275]] = true;
    local$$71799[_0x34b6[294]] = THREE[_0x34b6[1081]];
    local$$71799[_0x34b6[1092]] = THREE[_0x34b6[2132]];
    local$$71799[_0x34b6[1093]] = THREE[_0x34b6[2132]];
    local$$71799[_0x34b6[1820]] = THREE[_0x34b6[2133]];
    local$$71799[_0x34b6[1821]] = THREE[_0x34b6[2133]];
    /** @type {number} */
    local$$71799[_0x34b6[2134]] = 4;
    if (local$$71755[_0x34b6[1801]]) {
      window[_0x34b6[579]][_0x34b6[1819]](local$$71755[_0x34b6[1800]]);
    }
    local$$71755[_0x34b6[1800]] = _0x34b6[381];
    local$$71755[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
    local$$71756[_0x34b6[1755]]--;
  };
  var local$$71896 = new THREE.DDSLoader;
  var local$$71905 = local$$71896[_0x34b6[1060]](local$$71755[_0x34b6[1800]], local$$71757, local$$71762, local$$71797);
  return local$$71905;
};
/**
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1822]] = function() {
  if (this[_0x34b6[2121]][_0x34b6[1753]] > this[_0x34b6[2121]][_0x34b6[1752]]) {
    return;
  }
  /**
   * @param {?} local$$71936
   * @return {undefined}
   */
  var local$$71955 = function(local$$71936) {
    local$$71938[_0x34b6[1816]](LSELoadStatus.LS_NET_LOADED);
    local$$71938[_0x34b6[2121]][_0x34b6[1753]]--;
  };
  this[_0x34b6[1816]](LSELoadStatus.LS_NET_LOADING);
  /** @type {!XMLHttpRequest} */
  var local$$71964 = new XMLHttpRequest;
  local$$71964[_0x34b6[452]](_0x34b6[1044], this[_0x34b6[1785]], true);
  local$$71964[_0x34b6[1823]] = _0x34b6[1824];
  this[_0x34b6[2121]][_0x34b6[1753]]++;
  local$$71964[_0x34b6[1049]](null);
  /**
   * @param {?} local$$72001
   * @return {undefined}
   */
  local$$71964[_0x34b6[2135]] = function(local$$72001) {
    if (local$$72001[_0x34b6[1709]] && onProgressInfo != undefined) {
      /** @type {number} */
      var local$$72017 = local$$72001[_0x34b6[1710]] / local$$72001[_0x34b6[1711]] * 100;
      onProgressInfo(local$$72017);
    }
  };
  var local$$71938 = this;
  /**
   * @return {undefined}
   */
  local$$71964[_0x34b6[1236]] = function() {
    if (local$$71964[_0x34b6[1237]] == 4) {
      if (local$$71964[_0x34b6[1045]] == 200) {
        local$$71938[_0x34b6[1812]] = local$$71964[_0x34b6[1825]];
      } else {
      }
      local$$71938[_0x34b6[1816]](LSELoadStatus.LS_NET_LOADED);
      local$$71938[_0x34b6[2121]][_0x34b6[1753]]--;
      getScene()[_0x34b6[1720]]--;
    }
  };
};
/**
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1060]] = function() {
  if (this[_0x34b6[2121]][_0x34b6[1757]] > this[_0x34b6[2121]][_0x34b6[1756]]) {
    return;
  }
  if (this[_0x34b6[1812]] == null) {
    this[_0x34b6[1816]](LSELoadStatus.LS_LOADED);
    return;
  }
  this[_0x34b6[1816]](LSELoadStatus.LS_LOADING);
  var local$$72129 = this;
  this[_0x34b6[2121]][_0x34b6[1757]]++;
  /** @type {!Worker} */
  var local$$72142 = new Worker(_0x34b6[2136]);
  /**
   * @param {?} local$$72147
   * @return {undefined}
   */
  local$$72142[_0x34b6[1827]] = function(local$$72147) {
    var local$$72152 = local$$72147[_0x34b6[575]];
    if (local$$72152 != null && local$$72152 != undefined) {
      var local$$72158;
      {
        var local$$72163 = local$$72129[_0x34b6[1785]];
        local$$72158 = local$$72163[_0x34b6[702]](0, local$$72163[_0x34b6[382]](_0x34b6[1161]) + 1);
      }
      var local$$72187 = local$$72152[_0x34b6[2137]][_0x34b6[223]];
      /** @type {number} */
      var local$$72190 = 0;
      for (; local$$72190 < local$$72187; local$$72190++) {
        var local$$72196 = new LSJModelNodeTexture;
        var local$$72202 = local$$72152[_0x34b6[2137]][local$$72190];
        local$$72196[_0x34b6[2118]] = local$$72202[_0x34b6[2118]];
        if (local$$72202[_0x34b6[1800]] != _0x34b6[381]) {
          if (local$$72202[_0x34b6[1828]]) {
            local$$72196[_0x34b6[1800]] = LSJUtility[_0x34b6[1373]](local$$72158, local$$72202[_0x34b6[1800]]);
          } else {
            local$$72196[_0x34b6[1800]] = local$$72202[_0x34b6[1800]];
          }
        } else {
          if (local$$72202[_0x34b6[1829]] != null) {
            local$$72196[_0x34b6[1800]] = window[_0x34b6[579]][_0x34b6[1830]](local$$72202[_0x34b6[1829]]);
            /** @type {null} */
            local$$72202[_0x34b6[1829]] = null;
            /** @type {boolean} */
            local$$72196[_0x34b6[1801]] = true;
          }
        }
        if (local$$72196[_0x34b6[1800]] == _0x34b6[381] || local$$72196[_0x34b6[1800]] === undefined) {
          local$$72196[_0x34b6[1045]] = LSELoadStatus[_0x34b6[1786]];
        }
        local$$72129[_0x34b6[2122]][_0x34b6[220]](local$$72196);
      }
      var local$$72321 = local$$72152[_0x34b6[1810]][_0x34b6[223]];
      /** @type {number} */
      local$$72190 = 0;
      for (; local$$72190 < local$$72321; local$$72190++) {
        var local$$72333 = local$$72152[_0x34b6[1810]][local$$72190];
        var local$$72336 = new LSJModelNodeMaterial;
        var local$$72340 = new THREE.MeshPhongMaterial;
        local$$72336[_0x34b6[268]] = local$$72340;
        if (local$$72333[_0x34b6[2138]][0] != 0 || local$$72333[_0x34b6[2138]][1] != 0 || local$$72333[_0x34b6[2138]][2] != 0) {
          local$$72340[_0x34b6[245]] = (new THREE.Color)[_0x34b6[1533]](local$$72333[_0x34b6[2138]][0] / 255, local$$72333[_0x34b6[2138]][1] / 255, local$$72333[_0x34b6[2138]][2] / 255);
        }
        local$$72340[_0x34b6[1094]] = (new THREE.Color)[_0x34b6[1533]](local$$72333[_0x34b6[1094]][0] / 255, local$$72333[_0x34b6[1094]][1] / 255, local$$72333[_0x34b6[1094]][2] / 255);
        local$$72340[_0x34b6[2139]] = (new THREE.Color)[_0x34b6[1533]](local$$72333[_0x34b6[2139]][0] / 255, local$$72333[_0x34b6[2139]][1] / 255, local$$72333[_0x34b6[2139]][2] / 255);
        local$$72340[_0x34b6[1101]] = local$$72333[_0x34b6[1101]];
        if (local$$72333[_0x34b6[2138]][3] < 255) {
          /** @type {number} */
          local$$72340[_0x34b6[322]] = local$$72333[_0x34b6[2138]][3] / 255;
          /** @type {boolean} */
          local$$72340[_0x34b6[480]] = true;
        }
        var local$$72511 = local$$72333[_0x34b6[2119]][_0x34b6[223]];
        /** @type {number} */
        var local$$72514 = 0;
        for (; local$$72514 < local$$72511; local$$72514++) {
          local$$72336[_0x34b6[2119]][_0x34b6[220]](local$$72333[_0x34b6[2119]][local$$72514]);
        }
        local$$72129[_0x34b6[1810]][_0x34b6[220]](local$$72336);
      }
      local$$72129[_0x34b6[768]](local$$72152, local$$72129[_0x34b6[2122]], local$$72129[_0x34b6[1810]], local$$72158);
    }
    /** @type {null} */
    local$$72152 = null;
    /** @type {null} */
    local$$72147[_0x34b6[575]] = null;
    /** @type {null} */
    local$$72129[_0x34b6[1812]] = null;
    local$$72129[_0x34b6[1816]](LSELoadStatus.LS_LOADED);
    var local$$72582 = new THREE.Matrix4;
    var local$$72586 = new THREE.Quaternion;
    var local$$72590 = new THREE.Euler;
    local$$72590[_0x34b6[1882]] = _0x34b6[1560];
    local$$72590[_0x34b6[290]] = local$$72129[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2140]]()[_0x34b6[290]];
    local$$72590[_0x34b6[291]] = local$$72129[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2140]]()[_0x34b6[291]];
    local$$72590[_0x34b6[1287]] = local$$72129[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2140]]()[_0x34b6[1287]];
    local$$72586[_0x34b6[1554]](local$$72590);
    local$$72582[_0x34b6[2142]](local$$72129[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1240]](), local$$72586, local$$72129[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2141]]());
    local$$72129[_0x34b6[1762]][_0x34b6[1759]][_0x34b6[1358]](local$$72582);
    local$$72129[_0x34b6[1762]][_0x34b6[1806]][_0x34b6[1358]](local$$72582);
    /** @type {boolean} */
    local$$72129[_0x34b6[2125]] = true;
    local$$72129[_0x34b6[2121]][_0x34b6[1757]]--;
  };
  /**
   * @param {?} local$$72726
   * @return {undefined}
   */
  local$$72142[_0x34b6[556]] = function(local$$72726) {
    console[_0x34b6[514]](_0x34b6[1834] + local$$72726[_0x34b6[852]]);
    /** @type {null} */
    local$$72129[_0x34b6[1812]] = null;
    local$$72129[_0x34b6[1816]](LSELoadStatus.LS_LOADED);
    local$$72129[_0x34b6[2121]][_0x34b6[1757]]--;
  };
  local$$72142[_0x34b6[1835]](this[_0x34b6[1812]]);
};
/**
 * @param {!Array} local$$72781
 * @param {?} local$$72782
 * @param {?} local$$72783
 * @param {?} local$$72784
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[768]] = function(local$$72781, local$$72782, local$$72783, local$$72784) {
  if (local$$72781 == null || local$$72781 === undefined) {
    return;
  }
  /** @type {number} */
  var local$$72795 = 0;
  var local$$72803 = local$$72781[_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$72795 = 0;
  for (; local$$72795 < local$$72803; local$$72795++) {
    var local$$72812 = new LSJModelLODNode;
    this[_0x34b6[1760]](local$$72812);
    local$$72812[_0x34b6[768]](local$$72781[_0x34b6[684]][local$$72795], local$$72782, local$$72783, local$$72784);
  }
  this[_0x34b6[1808]] = local$$72781[_0x34b6[1808]];
  if (local$$72781[_0x34b6[1803]][_0x34b6[223]] > 0) {
    /** @type {number} */
    local$$72803 = local$$72781[_0x34b6[1803]][_0x34b6[223]] / 2;
    /** @type {number} */
    local$$72795 = 0;
    for (; local$$72795 < local$$72803; local$$72795++) {
      var local$$72865 = new THREE.Vector2;
      local$$72865[_0x34b6[290]] = local$$72781[_0x34b6[1803]][2 * local$$72795];
      local$$72865[_0x34b6[291]] = local$$72781[_0x34b6[1803]][2 * local$$72795 + 1];
      this[_0x34b6[1803]][_0x34b6[220]](local$$72865);
    }
  }
  if (this[_0x34b6[1785]] == _0x34b6[381] || this[_0x34b6[1785]] === undefined) {
    if (local$$72781[_0x34b6[1785]] != _0x34b6[381]) {
      this[_0x34b6[1785]] = local$$72784 + local$$72781[_0x34b6[1785]];
    }
  }
  if (local$$72781[_0x34b6[1759]][_0x34b6[223]] > 0) {
    this[_0x34b6[1759]] = new THREE.Sphere;
    var local$$72952 = new THREE.Vector3;
    local$$72952[_0x34b6[334]](local$$72781[_0x34b6[1759]][0], local$$72781[_0x34b6[1759]][1], local$$72781[_0x34b6[1759]][2]);
    if (local$$72952, local$$72781[_0x34b6[1759]][3] > 1.7E38) {
      local$$72952;
      /** @type {number} */
      local$$72781[_0x34b6[1759]][3] = 0;
    }
    this[_0x34b6[1759]][_0x34b6[334]](local$$72952, local$$72781[_0x34b6[1759]][3]);
    LSJMath[_0x34b6[1451]](this[_0x34b6[2121]][_0x34b6[2143]][_0x34b6[1759]], this[_0x34b6[1759]]);
  }
  if (local$$72781[_0x34b6[1806]][_0x34b6[223]] > 0) {
    this[_0x34b6[1759]] = new THREE.Sphere;
    var local$$73044 = new THREE.Vector3;
    local$$73044[_0x34b6[334]](local$$72781[_0x34b6[1806]][0], local$$72781[_0x34b6[1806]][1], local$$72781[_0x34b6[1806]][2]);
    var local$$73068 = new THREE.Vector3;
    local$$73068[_0x34b6[334]](local$$72781[_0x34b6[1806]][3], local$$72781[_0x34b6[1806]][4], local$$72781[_0x34b6[1806]][5]);
    this[_0x34b6[1806]][_0x34b6[334]](local$$73044, local$$73068);
    this[_0x34b6[2121]][_0x34b6[2143]][_0x34b6[1806]][_0x34b6[2144]](this[_0x34b6[1806]]);
  }
  var local$$73124 = local$$72781[_0x34b6[1836]][_0x34b6[223]];
  /** @type {number} */
  var local$$73127 = 0;
  for (; local$$73127 < local$$73124; local$$73127++) {
    var local$$73136 = local$$72781[_0x34b6[1836]][local$$73127];
    if (local$$73136[_0x34b6[1837]] != null) {
      var local$$73145 = new THREE.BufferGeometry;
      if (local$$73136[_0x34b6[1839]] != null) {
        local$$73145[_0x34b6[1355]](new THREE.BufferAttribute(local$$73136[_0x34b6[1839]], 1));
      }
      if (local$$73136[_0x34b6[1837]] != null) {
        local$$73145[_0x34b6[1174]](_0x34b6[430], new THREE.BufferAttribute(local$$73136[_0x34b6[1837]], 3));
      }
      if (local$$73136[_0x34b6[1129]] != null) {
        local$$73145[_0x34b6[1174]](_0x34b6[570], new THREE.BufferAttribute(local$$73136[_0x34b6[1129]], 3));
      }
      if (local$$73136[_0x34b6[674]] != null) {
        local$$73145[_0x34b6[1174]](_0x34b6[245], new THREE.BufferAttribute(local$$73136[_0x34b6[674]], local$$73136[_0x34b6[1840]]));
      }
      var local$$73237 = local$$73136[_0x34b6[1130]][_0x34b6[223]];
      /** @type {number} */
      k = 0;
      for (; k < local$$73237; k++) {
        if (local$$73136[_0x34b6[1130]][k] != null && local$$73136[_0x34b6[1130]][k] != undefined) {
          local$$73145[_0x34b6[1174]](_0x34b6[1176], new THREE.BufferAttribute(local$$73136[_0x34b6[1130]][k], 2));
        }
      }
      var local$$73282 = local$$73136[_0x34b6[2145]][_0x34b6[223]];
      /** @type {number} */
      var local$$73285 = -1;
      if (local$$73282 > 0) {
        local$$73145[_0x34b6[1355]](new THREE.BufferAttribute(local$$73136[_0x34b6[2145]][0][_0x34b6[1839]], 1));
        local$$73285 = local$$73136[_0x34b6[2145]][0][_0x34b6[1838]];
      }
      if (local$$73285 < 0 || local$$73285 >= local$$72783[_0x34b6[223]]) {
        local$$73285 = local$$73136[_0x34b6[1838]];
      }
      /** @type {null} */
      var local$$73334 = null;
      /** @type {null} */
      var local$$73337 = null;
      if (local$$73285 >= 0 && local$$73285 < local$$72783[_0x34b6[223]]) {
        local$$73334 = local$$72783[local$$73136[_0x34b6[1838]]];
        local$$73337 = local$$73334[_0x34b6[268]];
      }
      if (local$$73337 == null) {
        local$$73337 = lmModelDefaultMat;
      }
      var local$$73369 = new THREE.Mesh(local$$73145, local$$73337);
      if (local$$73136[_0x34b6[674]] != null && local$$73337 != null) {
        local$$73337[_0x34b6[1472]] = THREE[_0x34b6[1524]];
      }
      /** @type {boolean} */
      local$$73369[_0x34b6[1714]] = true;
      /** @type {boolean} */
      local$$73369[_0x34b6[1715]] = true;
      /** @type {number} */
      var local$$73403 = 0;
      if (local$$73334 != null) {
        this[_0x34b6[1811]][_0x34b6[220]](local$$73334);
        local$$73403 = local$$73334[_0x34b6[2119]][_0x34b6[223]];
      }
      /** @type {number} */
      var local$$73427 = 0;
      for (; local$$73427 < local$$73403; local$$73427++) {
        var local$$73439 = local$$73334[_0x34b6[2119]][local$$73427][_0x34b6[2146]];
        if (local$$73439 >= 0 && local$$73439 < local$$72782[_0x34b6[223]]) {
          this[_0x34b6[2123]][_0x34b6[220]](local$$72782[local$$73439]);
          if (local$$73334[_0x34b6[268]][_0x34b6[645]] == null || local$$73334[_0x34b6[268]][_0x34b6[645]] === undefined) {
            if (local$$72782[local$$73439][_0x34b6[276]] == null || local$$72782[local$$73439][_0x34b6[276]] === undefined) {
              local$$72782[local$$73439][_0x34b6[276]] = new THREE.Texture;
            }
            local$$73334[_0x34b6[268]][_0x34b6[645]] = local$$72782[local$$73439][_0x34b6[276]];
            if (local$$72782[local$$73439][_0x34b6[2118]]) {
              /** @type {boolean} */
              local$$73334[_0x34b6[268]][_0x34b6[480]] = true;
              /** @type {number} */
              local$$73334[_0x34b6[268]][_0x34b6[1273]] = .01;
            }
          }
        }
      }
      this[_0x34b6[1446]][_0x34b6[274]](local$$73369);
      this[_0x34b6[1818]](true);
    }
  }
  if (this[_0x34b6[1785]] == _0x34b6[381]) {
    this[_0x34b6[1807]] = LSELoadStatus[_0x34b6[1786]];
  }
};
/**
 * @param {?} local$$73584
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1841]] = function(local$$73584) {
  this[_0x34b6[1814]](false);
  var local$$73599 = this[_0x34b6[2121]][_0x34b6[1766]]();
  if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
    if (!local$$73599[_0x34b6[1842]](this[_0x34b6[1759]])) {
      return false;
    }
  }
  this[_0x34b6[1814]](true);
  return true;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1796]] = function() {
  if (this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1741]] && this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1845]] && this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
    return false;
  }
  if (this[_0x34b6[1846]]()) {
    return false;
  }
  /** @type {number} */
  var local$$73684 = 0;
  var local$$73692 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$73684 < local$$73692; local$$73684++) {
    if (!this[_0x34b6[684]][local$$73684][_0x34b6[1796]]()) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[2147]] = function() {
  /** @type {number} */
  var local$$73727 = 0;
  var local$$73735 = this[_0x34b6[2123]][_0x34b6[223]];
  for (; local$$73727 < local$$73735; local$$73727++) {
    if (this[_0x34b6[2123]][local$$73727][_0x34b6[1045]] != LSELoadStatus[_0x34b6[1786]]) {
      return false;
    }
  }
  return true;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[2148]] = function() {
  /** @type {number} */
  var local$$73772 = 0;
  var local$$73780 = this[_0x34b6[2123]][_0x34b6[223]];
  for (; local$$73772 < local$$73780; local$$73772++) {
    if (this[_0x34b6[2123]][local$$73772][_0x34b6[1045]] != LSELoadStatus[_0x34b6[1741]] && this[_0x34b6[2123]][local$$73772][_0x34b6[1045]] != LSELoadStatus[_0x34b6[1786]]) {
      return true;
    }
  }
  return false;
};
/**
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1848]] = function() {
  /** @type {number} */
  var local$$73829 = 0;
  if (this[_0x34b6[1817]]()) {
    /** @type {number} */
    local$$73829 = local$$73829 + 1;
  }
  /** @type {number} */
  var local$$73843 = 0;
  var local$$73851 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$73843 < local$$73851; local$$73843++) {
    local$$73829 = local$$73829 + this[_0x34b6[684]][local$$73843][_0x34b6[1848]]();
  }
  return local$$73829;
};
/**
 * @return {undefined}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1797]] = function() {
  /** @type {number} */
  var local$$73883 = 0;
  /** @type {number} */
  var local$$73886 = 0;
  var local$$73894 = this[_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$73886 = 0;
  for (; local$$73886 < local$$73894; local$$73886++) {
    this[_0x34b6[684]][local$$73886][_0x34b6[1797]]();
  }
  this[_0x34b6[684]][_0x34b6[222]](0, local$$73894);
  this[_0x34b6[1803]][_0x34b6[222]](0, this[_0x34b6[1803]][_0x34b6[223]]);
  this[_0x34b6[1811]][_0x34b6[222]](0, this[_0x34b6[1811]][_0x34b6[223]]);
  this[_0x34b6[2123]][_0x34b6[222]](0, this[_0x34b6[2123]][_0x34b6[223]]);
  /** @type {number} */
  var local$$73979 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]] - 1;
  for (; local$$73979 >= 0; local$$73979--) {
    var local$$73992 = this[_0x34b6[1446]][_0x34b6[684]][local$$73979];
    this[_0x34b6[1446]][_0x34b6[1448]](local$$73992);
    if (local$$73992 != null && local$$73992 instanceof THREE[_0x34b6[329]]) {
      if (local$$73992[_0x34b6[1126]]) {
        local$$73992[_0x34b6[1126]][_0x34b6[232]]();
      }
      /** @type {null} */
      local$$73992[_0x34b6[268]] = null;
      /** @type {null} */
      local$$73992[_0x34b6[1126]] = null;
      this[_0x34b6[2121]][_0x34b6[1773]](1);
    }
    /** @type {null} */
    local$$73992 = null;
  }
  local$$73894 = this[_0x34b6[1810]][_0x34b6[223]];
  /** @type {number} */
  local$$73886 = 0;
  for (; local$$73886 < local$$73894; local$$73886++) {
    var local$$74071 = this[_0x34b6[1810]][local$$73886];
    if (local$$74071[_0x34b6[268]] != null && local$$74071[_0x34b6[268]] != undefined) {
      local$$74071[_0x34b6[268]][_0x34b6[232]]();
      /** @type {null} */
      local$$74071[_0x34b6[268]][_0x34b6[645]] = null;
      /** @type {null} */
      local$$74071[_0x34b6[268]] = null;
    }
  }
  this[_0x34b6[1810]][_0x34b6[222]](0, local$$73894);
  local$$73894 = this[_0x34b6[2122]][_0x34b6[223]];
  /** @type {number} */
  local$$73886 = 0;
  for (; local$$73886 < local$$73894; local$$73886++) {
    var local$$74138 = this[_0x34b6[2122]][local$$73886];
    if (local$$74138[_0x34b6[276]] != null && local$$74138[_0x34b6[276]] != undefined) {
      local$$74138[_0x34b6[276]][_0x34b6[232]]();
      /** @type {null} */
      local$$74138[_0x34b6[276]][_0x34b6[554]] = null;
      /** @type {null} */
      local$$74138[_0x34b6[276]] = null;
    }
  }
  this[_0x34b6[2122]][_0x34b6[222]](0, local$$73894);
  /** @type {boolean} */
  this[_0x34b6[1809]] = false;
  /** @type {null} */
  this[_0x34b6[1812]] = null;
  this[_0x34b6[1816]](LSELoadStatus.LS_UNLOAD);
};
/**
 * @param {?} local$$74215
 * @return {?}
 */
LSJModelLODNode[_0x34b6[219]][_0x34b6[1850]] = function(local$$74215) {
  /** @type {number} */
  var local$$74218 = 0;
  var local$$74226 = this[_0x34b6[684]][_0x34b6[223]];
  for (; local$$74218 < local$$74226; local$$74218++) {
    var local$$74235 = this[_0x34b6[684]][local$$74218];
    if (local$$74235 != null) {
      if (local$$74235[_0x34b6[1841]](local$$74215) && local$$74235[_0x34b6[684]][_0x34b6[223]] > 1) {
        local$$74235[_0x34b6[1814]](true);
        var local$$74263 = local$$74235[_0x34b6[684]][0];
        if (local$$74263) {
          if (local$$74263[_0x34b6[1785]] != _0x34b6[381]) {
            if (local$$74263[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
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
LSJModelLODNode[_0x34b6[219]][_0x34b6[1261]] = function(local$$74306) {
  /** @type {boolean} */
  this[_0x34b6[1446]][_0x34b6[330]] = false;
  var local$$74326 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  var local$$74329 = 0;
  /** @type {number} */
  local$$74329 = 0;
  for (; local$$74329 < local$$74326; local$$74329++) {
    /** @type {boolean} */
    this[_0x34b6[1446]][_0x34b6[684]][local$$74329][_0x34b6[330]] = false;
  }
  /** @type {boolean} */
  this[_0x34b6[1804]] = false;
  {
    var local$$74361 = new THREE.Matrix4;
    var local$$74365 = new THREE.Quaternion;
    var local$$74369 = new THREE.Euler;
    local$$74369[_0x34b6[1882]] = _0x34b6[1560];
    local$$74369[_0x34b6[290]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2140]]()[_0x34b6[290]];
    local$$74369[_0x34b6[291]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2140]]()[_0x34b6[291]];
    local$$74369[_0x34b6[1287]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2140]]()[_0x34b6[1287]];
    local$$74365[_0x34b6[1554]](local$$74369);
    local$$74361[_0x34b6[2142]](this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1240]](), local$$74365, this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2141]]());
    local$$74326 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]];
    /** @type {number} */
    local$$74329 = 0;
    for (; local$$74329 < local$$74326; local$$74329++) {
      var local$$74486 = this[_0x34b6[1446]][_0x34b6[684]][local$$74329];
      if (local$$74486 && local$$74486 instanceof THREE[_0x34b6[329]]) {
        local$$74486[_0x34b6[1090]][_0x34b6[290]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1090]][_0x34b6[290]];
        local$$74486[_0x34b6[1090]][_0x34b6[291]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1090]][_0x34b6[291]];
        local$$74486[_0x34b6[1090]][_0x34b6[1287]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1090]][_0x34b6[1287]];
        local$$74486[_0x34b6[430]][_0x34b6[290]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1240]]()[_0x34b6[290]];
        local$$74486[_0x34b6[430]][_0x34b6[291]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1240]]()[_0x34b6[291]];
        local$$74486[_0x34b6[430]][_0x34b6[1287]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1240]]()[_0x34b6[1287]];
        local$$74486[_0x34b6[1271]][_0x34b6[290]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1598]][_0x34b6[290]];
        local$$74486[_0x34b6[1271]][_0x34b6[291]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1598]][_0x34b6[291]];
        local$$74486[_0x34b6[1271]][_0x34b6[1287]] = this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[1598]][_0x34b6[1287]];
        /** @type {boolean} */
        local$$74486[_0x34b6[2125]] = true;
        this[_0x34b6[1759]][_0x34b6[1358]](local$$74361);
        this[_0x34b6[1806]][_0x34b6[1358]](local$$74361);
      }
    }
    /** @type {boolean} */
    this[_0x34b6[2125]] = false;
  }
  if (!this[_0x34b6[1841]](local$$74306)) {
    /** @type {boolean} */
    this[_0x34b6[1446]][_0x34b6[330]] = false;
    return false;
  }
  this[_0x34b6[1769]](this[_0x34b6[2121]][_0x34b6[1770]]());
  this[_0x34b6[1771]](this[_0x34b6[2121]][_0x34b6[1772]]());
  if (this[_0x34b6[1785]] != _0x34b6[381]) {
    if (this[_0x34b6[1792]]() == LSELoadStatus[_0x34b6[1741]]) {
      getScene()[_0x34b6[1720]]++;
      this[_0x34b6[1822]]();
    }
    if (this[_0x34b6[1792]]() == LSELoadStatus[_0x34b6[1845]]) {
      this[_0x34b6[1060]]();
    }
    if (this[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
      this[_0x34b6[2121]][_0x34b6[1758]]++;
      return false;
    }
  }
  /** @type {number} */
  var local$$74821 = 0;
  if (this[_0x34b6[1803]][_0x34b6[223]] > 0) {
    if (this[_0x34b6[1808]] == LSERangeMode[_0x34b6[1851]]) {
      if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
        local$$74821 = LSJMath[_0x34b6[1700]](this[_0x34b6[1759]][_0x34b6[658]], this[_0x34b6[2121]][_0x34b6[1765]]());
      }
    } else {
      if (this[_0x34b6[1808]] == LSERangeMode[_0x34b6[1852]]) {
        if (!this[_0x34b6[1759]][_0x34b6[1456]]()) {
          local$$74821 = LSJMath[_0x34b6[1705]](this[_0x34b6[1759]], this[_0x34b6[2121]][_0x34b6[1763]]());
        }
      }
    }
  }
  /** @type {boolean} */
  var local$$74909 = true;
  /** @type {number} */
  var local$$74912 = 0;
  local$$74326 = this[_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$74329 = 0;
  for (; local$$74329 < local$$74326; local$$74329++) {
    var local$$74932 = this[_0x34b6[684]][local$$74329];
    if (local$$74912 < this[_0x34b6[1803]][_0x34b6[223]]) {
      var local$$74945 = this[_0x34b6[1803]][local$$74912];
      if (local$$74932 && local$$74821 >= local$$74945[_0x34b6[290]] && local$$74821 < local$$74945[_0x34b6[291]]) {
        if (local$$74932[_0x34b6[1261]](local$$74306)) {
          /** @type {boolean} */
          this[_0x34b6[1804]] = true;
        } else {
          if (local$$74932[_0x34b6[1815]]()) {
            /** @type {number} */
            var local$$74974 = local$$74329 - 1;
            for (; local$$74974 >= 0; local$$74974--) {
              if (this[_0x34b6[684]][local$$74974][_0x34b6[1261]](local$$74306)) {
                /** @type {boolean} */
                this[_0x34b6[1804]] = true;
                break;
              }
            }
          }
        }
      }
      local$$74912++;
    } else {
      if (local$$74932 && local$$74909) {
        if (local$$74932[_0x34b6[1261]](local$$74306)) {
          /** @type {boolean} */
          this[_0x34b6[1804]] = true;
        }
      }
    }
  }
  /** @type {boolean} */
  this[_0x34b6[1446]][_0x34b6[330]] = true;
  /** @type {boolean} */
  var local$$75039 = false;
  var local$$75045 = this[_0x34b6[2147]]();
  if (!this[_0x34b6[1804]] && this[_0x34b6[2121]][_0x34b6[1755]] < this[_0x34b6[2121]][_0x34b6[1754]]) {
    local$$74326 = this[_0x34b6[2123]][_0x34b6[223]];
    /** @type {number} */
    local$$74329 = 0;
    for (; local$$74329 < local$$74326; local$$74329++) {
      var local$$75083 = this[_0x34b6[2123]][local$$74329];
      if (local$$75083[_0x34b6[1045]] == LSELoadStatus[_0x34b6[1741]]) {
        this[_0x34b6[1089]](local$$75083, this[_0x34b6[2121]]);
      }
    }
  }
  if (local$$75045 && !this[_0x34b6[2124]] || this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2149]]) {
    /** @type {number} */
    local$$74329 = 0;
    for (; local$$74329 < this[_0x34b6[1811]][_0x34b6[223]]; local$$74329++) {
      var local$$75138 = this[_0x34b6[1811]][local$$74329];
      local$$75083 = this[_0x34b6[2123]][local$$74329];
      if (local$$75083 != undefined) {
        local$$75138[_0x34b6[268]][_0x34b6[245]] = (new THREE.Color)[_0x34b6[1533]](1, 1, 1);
      }
      if (this[_0x34b6[1762]][_0x34b6[2121]][_0x34b6[2150]]) {
        local$$75138[_0x34b6[268]][_0x34b6[245]] = (new THREE.Color)[_0x34b6[1533]](.6, .6, 1);
      }
    }
    /** @type {boolean} */
    this[_0x34b6[2124]] = true;
  }
  local$$74326 = this[_0x34b6[1446]][_0x34b6[684]][_0x34b6[223]];
  /** @type {number} */
  local$$74329 = 0;
  for (; local$$74329 < local$$74326; local$$74329++) {
    local$$74486 = this[_0x34b6[1446]][_0x34b6[684]][local$$74329];
    if (local$$74486 && local$$74486 instanceof THREE[_0x34b6[329]]) {
      if (!this[_0x34b6[1804]] && local$$75045) {
        /** @type {boolean} */
        local$$74486[_0x34b6[330]] = true;
        /** @type {boolean} */
        local$$75039 = true;
      } else {
        /** @type {boolean} */
        local$$74486[_0x34b6[330]] = false;
      }
    }
  }
  if (!local$$75039) {
    this[_0x34b6[1446]][_0x34b6[330]] = this[_0x34b6[1804]];
    return this[_0x34b6[1804]];
  }
  /** @type {boolean} */
  this[_0x34b6[1804]] = true;
  return true;
};
/**
 * @return {undefined}
 */
LSJGeoModelLOD = function() {
  LSJGeometry[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1727];
  this[_0x34b6[2151]] = _0x34b6[381];
  this[_0x34b6[430]] = new THREE.Vector3(0, 0, 0);
  this[_0x34b6[1598]] = new THREE.Vector3(0, 0, 0);
  this[_0x34b6[1090]] = new THREE.Vector3(1, 1, 1);
  /** @type {number} */
  this[_0x34b6[1737]] = 0;
  /** @type {number} */
  this[_0x34b6[1738]] = 200;
  this[_0x34b6[2143]] = new LSJModelLODNode;
  this[_0x34b6[2143]][_0x34b6[2121]] = this;
  this[_0x34b6[2143]][_0x34b6[1762]] = this[_0x34b6[2143]];
  this[_0x34b6[1446]][_0x34b6[274]](this[_0x34b6[2143]][_0x34b6[1446]]);
  this[_0x34b6[1446]][_0x34b6[1988]] = this;
  this[_0x34b6[1743]] = new THREE.Frustum;
  this[_0x34b6[1744]] = new THREE.Vector4;
  this[_0x34b6[1745]] = new THREE.Matrix4;
  this[_0x34b6[1746]] = new THREE.Matrix4;
  this[_0x34b6[1747]] = new THREE.Matrix4;
  this[_0x34b6[1748]] = new THREE.Matrix4;
  this[_0x34b6[1749]] = new THREE.Vector4;
  /** @type {number} */
  this[_0x34b6[1750]] = 0;
  /** @type {number} */
  this[_0x34b6[1751]] = 0;
  /** @type {number} */
  this[_0x34b6[1752]] = 2;
  /** @type {number} */
  this[_0x34b6[1753]] = 0;
  /** @type {number} */
  this[_0x34b6[1754]] = 2;
  /** @type {number} */
  this[_0x34b6[1755]] = 0;
  /** @type {number} */
  this[_0x34b6[1756]] = 2;
  /** @type {number} */
  this[_0x34b6[1757]] = 0;
  /** @type {number} */
  this[_0x34b6[1758]] = 0;
  /** @type {boolean} */
  this[_0x34b6[2125]] = false;
};
LSJGeoModelLOD[_0x34b6[219]] = Object[_0x34b6[242]](LSJGeometry[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1183]] = LSJGeoModelLOD;
/**
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[232]] = function() {
  this[_0x34b6[2143]][_0x34b6[1797]]();
};
/**
 * @param {?} local$$75567
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2152]] = function(local$$75567) {
  if (this[_0x34b6[2151]] != local$$75567) {
    this[_0x34b6[232]]();
    var local$$75579 = new LSJModelLODNode;
    local$$75579[_0x34b6[1785]] = local$$75567;
    this[_0x34b6[2143]][_0x34b6[1760]](local$$75579);
    this[_0x34b6[2151]] = local$$75567;
  }
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2153]] = function() {
  return this[_0x34b6[2151]];
};
/**
 * @param {?} local$$75626
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1689]] = function(local$$75626) {
  this[_0x34b6[1115]] = local$$75626;
};
/**
 * @param {?} local$$75643
 * @param {?} local$$75644
 * @param {?} local$$75645
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1241]] = function(local$$75643, local$$75644, local$$75645) {
  this[_0x34b6[430]][_0x34b6[290]] = local$$75643;
  this[_0x34b6[430]][_0x34b6[291]] = local$$75644;
  this[_0x34b6[430]][_0x34b6[1287]] = local$$75645;
  /** @type {boolean} */
  this[_0x34b6[2125]] = true;
  this[_0x34b6[2143]][_0x34b6[2126]](true);
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1240]] = function() {
  return this[_0x34b6[430]];
};
/**
 * @param {?} local$$75711
 * @param {?} local$$75712
 * @param {?} local$$75713
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2154]] = function(local$$75711, local$$75712, local$$75713) {
  this[_0x34b6[1598]][_0x34b6[290]] = local$$75711;
  this[_0x34b6[1598]][_0x34b6[291]] = local$$75712;
  this[_0x34b6[1598]][_0x34b6[1287]] = local$$75713;
  /** @type {boolean} */
  this[_0x34b6[2125]] = true;
  this[_0x34b6[2143]][_0x34b6[2126]](true);
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2140]] = function() {
  return this[_0x34b6[1598]];
};
/**
 * @param {?} local$$75779
 * @param {?} local$$75780
 * @param {?} local$$75781
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2155]] = function(local$$75779, local$$75780, local$$75781) {
  this[_0x34b6[1090]][_0x34b6[290]] = local$$75779;
  this[_0x34b6[1090]][_0x34b6[291]] = local$$75780;
  this[_0x34b6[1090]][_0x34b6[1287]] = local$$75781;
  /** @type {boolean} */
  this[_0x34b6[2125]] = true;
  this[_0x34b6[2143]][_0x34b6[2126]](true);
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2141]] = function() {
  return this[_0x34b6[1090]];
};
/**
 * @param {?} local$$75847
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2156]] = function(local$$75847) {
  if (this[_0x34b6[2150]] != local$$75847) {
    this[_0x34b6[2150]] = local$$75847;
    /** @type {boolean} */
    this[_0x34b6[2149]] = true;
  }
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[2157]] = function() {
  return this[_0x34b6[2143]][_0x34b6[1806]];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1450]] = function() {
  return this[_0x34b6[2143]][_0x34b6[1759]];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1763]] = function() {
  return this[_0x34b6[1749]];
};
/**
 * @param {?} local$$75927
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1764]] = function(local$$75927) {
  this[_0x34b6[1749]] = local$$75927;
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1765]] = function() {
  return this[_0x34b6[1272]];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1766]] = function() {
  return this[_0x34b6[1743]];
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1767]] = function() {
  return this[_0x34b6[1744]];
};
/**
 * @param {?} local$$75989
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1768]] = function(local$$75989) {
  this[_0x34b6[1744]] = local$$75989;
};
/**
 * @param {?} local$$76006
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1769]] = function(local$$76006) {
  this[_0x34b6[1751]] = local$$76006;
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1770]] = function() {
  return this[_0x34b6[1751]];
};
/**
 * @param {?} local$$76038
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1771]] = function(local$$76038) {
  this[_0x34b6[1750]] = local$$76038;
};
/**
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1772]] = function() {
  return this[_0x34b6[1750]];
};
/**
 * @param {?} local$$76070
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1773]] = function(local$$76070) {
  this[_0x34b6[1737]] -= local$$76070;
};
/**
 * @param {?} local$$76087
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1774]] = function(local$$76087) {
  this[_0x34b6[1737]] += local$$76087;
};
/**
 * @param {!Object} local$$76104
 * @return {?}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1795]] = function(local$$76104) {
  if (this[_0x34b6[1737]] < this[_0x34b6[1738]]) {
    return false;
  }
  if (local$$76104 != null) {
    if (local$$76104[_0x34b6[1792]]() != LSELoadStatus[_0x34b6[1786]]) {
      return false;
    }
    /** @type {number} */
    var local$$76134 = 0;
    var local$$76142 = local$$76104[_0x34b6[684]][_0x34b6[223]];
    for (; local$$76134 < local$$76142; local$$76134++) {
      this[_0x34b6[1795]](local$$76104[_0x34b6[684]][local$$76134]);
    }
    if (this[_0x34b6[1737]] < this[_0x34b6[1738]]) {
      return false;
    }
    if (local$$76104[_0x34b6[1785]] == _0x34b6[381]) {
      return false;
    }
    /** @type {number} */
    var local$$76191 = this[_0x34b6[1772]]() - local$$76104[_0x34b6[1772]]();
    /** @type {number} */
    var local$$76202 = this[_0x34b6[1770]]() - local$$76104[_0x34b6[1770]]();
    if (local$$76191 < 5 || local$$76202 < 100) {
      return false;
    }
    if (local$$76104[_0x34b6[1796]]()) {
      local$$76104[_0x34b6[1797]]();
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
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1798]] = function(local$$76245) {
  local$$76245[_0x34b6[1263]]();
  var local$$76254 = new THREE.Matrix4;
  local$$76254[_0x34b6[1491]](local$$76245[_0x34b6[1285]]);
  this[_0x34b6[1747]][_0x34b6[1286]](local$$76254, this[_0x34b6[1745]]);
  this[_0x34b6[1748]][_0x34b6[1286]](local$$76245[_0x34b6[335]], this[_0x34b6[1747]]);
  this[_0x34b6[1743]][_0x34b6[1492]](this[_0x34b6[1748]]);
  this[_0x34b6[1749]] = LSJMath[_0x34b6[1706]](this[_0x34b6[1744]], local$$76245[_0x34b6[335]], this[_0x34b6[1747]]);
};
/**
 * @param {?} local$$76328
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[1261]] = function(local$$76328) {
  this[_0x34b6[1751]] = (new Date)[_0x34b6[1723]]();
  ++this[_0x34b6[1750]];
  this[_0x34b6[1798]](local$$76328);
  /** @type {number} */
  this[_0x34b6[1758]] = 0;
  this[_0x34b6[2143]][_0x34b6[1261]](local$$76328);
  if (this[_0x34b6[1737]] > this[_0x34b6[1738]]) {
    this[_0x34b6[1795]](this[_0x34b6[2143]]);
  }
};
/**
 * @param {?} local$$76391
 * @return {undefined}
 */
LSJGeoModelLOD[_0x34b6[219]][_0x34b6[225]] = function(local$$76391) {
  this[_0x34b6[1446]][_0x34b6[330]] = this[_0x34b6[330]];
  if (!this[_0x34b6[330]]) {
    return;
  }
  this[_0x34b6[1261]](local$$76391[_0x34b6[1692]]);
};
/**
 * @return {undefined}
 */
LSJFeatureLayer = function() {
  LSJLayer[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1458];
  this[_0x34b6[1739]] = _0x34b6[381];
  /** @type {!Array} */
  this[_0x34b6[2158]] = [];
};
LSJFeatureLayer[_0x34b6[219]] = Object[_0x34b6[242]](LSJLayer[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJFeatureLayer[_0x34b6[219]][_0x34b6[1183]] = LSJFeatureLayer;
/**
 * @return {undefined}
 */
LSJFeatureLayer[_0x34b6[219]][_0x34b6[232]] = function() {
};
/**
 * @return {?}
 */
LSJFeatureLayer[_0x34b6[219]][_0x34b6[1450]] = function() {
  return this[_0x34b6[1447]];
};
/**
 * @param {?} local$$76507
 * @return {undefined}
 */
LSJFeatureLayer[_0x34b6[219]][_0x34b6[1986]] = function(local$$76507) {
  local$$76507[_0x34b6[2156]](true);
  this[_0x34b6[2158]][_0x34b6[220]](local$$76507);
};
/**
 * @return {undefined}
 */
LSJFeatureLayer[_0x34b6[219]][_0x34b6[1457]] = function() {
  var local$$76540 = this[_0x34b6[2158]][_0x34b6[223]];
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
LSJFeatureLayer[_0x34b6[219]][_0x34b6[1059]] = function(local$$76576) {
  if (local$$76576 == _0x34b6[381]) {
    return;
  }
  this[_0x34b6[1739]] = local$$76576;
  var local$$76590 = this;
  var local$$76594 = new THREE.XHRLoader;
  local$$76594[_0x34b6[1060]](local$$76576, function(local$$76599) {
    var local$$76605 = JSON[_0x34b6[768]](local$$76599);
    var local$$76610 = local$$76605[_0x34b6[1788]];
    if (local$$76610 !== undefined) {
      if (local$$76610[_0x34b6[1789]] !== undefined) {
        var local$$76631 = new THREE.Vector3(local$$76610[_0x34b6[1789]].West, local$$76610[_0x34b6[1789]].South, local$$76610[_0x34b6[1789]].MinZ);
        var local$$76647 = new THREE.Vector3(local$$76610[_0x34b6[1789]].East, local$$76610[_0x34b6[1789]].North, local$$76610[_0x34b6[1789]].MaxZ);
        var local$$76651 = new THREE.Vector3;
        local$$76651[_0x34b6[334]](local$$76631[_0x34b6[290]] / 2 + local$$76647[_0x34b6[290]] / 2, local$$76631[_0x34b6[291]] / 2 + local$$76647[_0x34b6[291]] / 2, local$$76631[_0x34b6[1287]] / 2 + local$$76647[_0x34b6[1287]] / 2);
        var local$$76693 = new THREE.Vector3;
        local$$76693[_0x34b6[1697]](local$$76647, local$$76631);
        local$$76590[_0x34b6[1447]][_0x34b6[334]](local$$76651, local$$76693[_0x34b6[223]]() / 2);
      }
      var local$$76720 = local$$76610[_0x34b6[2159]];
      if (local$$76720 !== undefined) {
        var local$$76726 = local$$76720[_0x34b6[223]];
        /** @type {number} */
        var local$$76729 = 0;
        for (; local$$76729 < local$$76726; local$$76729++) {
          var local$$76735 = local$$76720[local$$76729];
          if (local$$76735 !== undefined) {
            var local$$76739 = new LSJGeoModelLOD;
            local$$76739[_0x34b6[1689]](local$$76735[_0x34b6[2160]].Name);
            var local$$76763 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2161]][_0x34b6[290]];
            var local$$76777 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2161]][_0x34b6[291]];
            var local$$76791 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2161]][_0x34b6[1287]];
            local$$76739[_0x34b6[1241]](local$$76763, local$$76777, local$$76791);
            local$$76739[_0x34b6[1729]] = local$$76590;
            var local$$76815 = local$$76735[_0x34b6[2160]][_0x34b6[2162]][_0x34b6[2163]][_0x34b6[549]];
            local$$76739[_0x34b6[2152]](LSJUtility[_0x34b6[1373]](LSJUtility[_0x34b6[1372]](local$$76576), local$$76815));
            local$$76590[_0x34b6[1728]](local$$76739);
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
  LSJLayer[_0x34b6[238]](this);
  /** @type {number} */
  this[_0x34b6[2164]] = 1;
  /** @type {number} */
  this[_0x34b6[322]] = 1;
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
LSJPointCloudLayer[_0x34b6[219]] = Object[_0x34b6[242]](LSJLayer[_0x34b6[219]]);
/** @type {function(): undefined} */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[1183]] = LSJPointCloudLayer;
/**
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[232]] = function() {
};
/**
 * @param {?} local$$76957
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2175]] = function(local$$76957) {
  if (this[_0x34b6[2164]] !== local$$76957) {
    this[_0x34b6[2164]] = local$$76957;
  }
};
/**
 * @return {?}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2176]] = function() {
  return this[_0x34b6[2164]];
};
/**
 * @param {?} local$$76995
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2177]] = function(local$$76995) {
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
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2182]] = function() {
  return this[_0x34b6[2165]];
};
/**
 * @param {?} local$$77083
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2183]] = function(local$$77083) {
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
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2189]] = function() {
  return this[_0x34b6[2171]];
};
/**
 * @param {?} local$$77170
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2190]] = function(local$$77170) {
  this[_0x34b6[2191]](local$$77170);
};
/**
 * @return {?}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2192]] = function() {
  return this[_0x34b6[2193]](this[_0x34b6[2170]]);
};
/**
 * @param {?} local$$77206
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2191]] = function(local$$77206) {
  if (local$$77206 === _0x34b6[2194]) {
    this[_0x34b6[2170]] = Potree[_0x34b6[2195]][_0x34b6[2194]];
  } else {
    if (local$$77206 === _0x34b6[871]) {
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
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2193]] = function(local$$77421) {
  if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2194]]) {
    return _0x34b6[2194];
  } else {
    if (local$$77421 === Potree[_0x34b6[2195]][_0x34b6[2196]]) {
      return _0x34b6[871];
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
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2217]] = function(local$$77588, local$$77589) {
  if (this[_0x34b6[2173]] !== local$$77588 || this[_0x34b6[2174]] !== local$$77589) {
    this[_0x34b6[2173]] = local$$77588 || this[_0x34b6[2173]];
    this[_0x34b6[2174]] = local$$77589 || this[_0x34b6[2174]];
  }
};
/**
 * @return {?}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[2218]] = function() {
  return {
    min : this[_0x34b6[2173]],
    max : this[_0x34b6[2174]]
  };
};
/**
 * @return {?}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[1450]] = function() {
  return this[_0x34b6[1447]];
};
/**
 * @param {?} local$$77664
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[1059]] = function(local$$77664) {
  var local$$77668 = new THREE.Object3D;
  this[_0x34b6[1446]][_0x34b6[274]](local$$77668);
  var local$$77678 = this;
  Potree[_0x34b6[2222]][_0x34b6[1060]](local$$77664, function(local$$77686) {
    local$$77678[_0x34b6[2219]] = new Potree.PointCloudOctree(local$$77686);
    local$$77678[_0x34b6[2219]][_0x34b6[268]][_0x34b6[2220]] = Potree[_0x34b6[2220]][_0x34b6[2221]];
    local$$77668[_0x34b6[274]](local$$77678[_0x34b6[2219]]);
    local$$77668[_0x34b6[1263]](true);
    local$$77678[_0x34b6[1447]] = local$$77678[_0x34b6[2219]][_0x34b6[1447]][_0x34b6[212]]()[_0x34b6[1358]](local$$77678[_0x34b6[2219]][_0x34b6[1285]]);
  });
};
/**
 * @param {?} local$$77766
 * @return {undefined}
 */
LSJPointCloudLayer[_0x34b6[219]][_0x34b6[225]] = function(local$$77766) {
  if (this[_0x34b6[2219]] != undefined) {
    this[_0x34b6[2219]][_0x34b6[268]][_0x34b6[1573]] = this[_0x34b6[2164]];
    this[_0x34b6[2219]][_0x34b6[268]][_0x34b6[322]] = this[_0x34b6[322]];
    this[_0x34b6[2219]][_0x34b6[268]][_0x34b6[2170]] = this[_0x34b6[2170]];
    this[_0x34b6[2219]][_0x34b6[268]][_0x34b6[2167]] = this[_0x34b6[2167]];
    this[_0x34b6[2219]][_0x34b6[268]][_0x34b6[2173]] = this[_0x34b6[2173]];
    this[_0x34b6[2219]][_0x34b6[268]][_0x34b6[2174]] = this[_0x34b6[2174]];
    this[_0x34b6[2219]][_0x34b6[1261]](controlCamera, controlRender);
  }
};
/**
 * @param {?} local$$77873
 * @return {undefined}
 */
LSJGeoLabel = function(local$$77873) {
  LSJGeometry[_0x34b6[238]](this);
  this[_0x34b6[445]] = _0x34b6[1290];
  this[_0x34b6[430]] = new THREE.Vector3(0, 0, 0);
  this[_0x34b6[1684]] = undefined;
  /** @type {boolean} */
  this[_0x34b6[1685]] = true;
  this[_0x34b6[1686]] = new LSJRectangle(0, 0, 0, 0);
  /** @type {number} */
  this[_0x34b6[1688]] = 1;
  this[_0x34b6[268]] = undefined;
  this[_0x34b6[243]] = _0x34b6[381];
  /** @type {number} */
  this[_0x34b6[2223]] = 100;
  /** @type {number} */
  this[_0x34b6[1349]] = 0;
  /** @type {number} */
  this[_0x34b6[1350]] = 0;
  this[_0x34b6[2224]] = local$$77873;
  this[_0x34b6[1261]]();
};
LSJGeoLabel[_0x34b6[219]] = Object[_0x34b6[242]](LSJGeometry[_0x34b6[219]]);
/** @type {function(?): undefined} */
LSJGeoLabel[_0x34b6[219]][_0x34b6[1183]] = LSJGeoLabel;
/**
 * @param {?} local$$77994
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[2225]] = function(local$$77994) {
  this[_0x34b6[1349]] = local$$77994;
};
/**
 * @return {?}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[2226]] = function() {
  return this[_0x34b6[1349]];
};
/**
 * @return {?}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[2227]] = function() {
  return this[_0x34b6[1350]];
};
/**
 * @param {?} local$$78041
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[2228]] = function(local$$78041) {
  this[_0x34b6[1350]] = local$$78041;
};
/**
 * @param {?} local$$78058
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[221]] = function(local$$78058) {
  this[_0x34b6[2223]] = local$$78058;
};
/**
 * @return {?}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[207]] = function() {
  return this[_0x34b6[2223]];
};
/**
 * @param {?} local$$78090
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[2229]] = function(local$$78090) {
  this[_0x34b6[2224]] = local$$78090;
  /** @type {boolean} */
  this[_0x34b6[1685]] = true;
};
/**
 * @param {?} local$$78113
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[1689]] = function(local$$78113) {
  this[_0x34b6[1115]] = local$$78113;
};
/**
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[1261]] = function() {
  if (this[_0x34b6[243]] == _0x34b6[2230]) {
    return;
  }
  this[_0x34b6[232]]();
  var local$$78146 = this;
  this[_0x34b6[243]] = _0x34b6[2230];
  this[_0x34b6[2224]][_0x34b6[428]][_0x34b6[687]] = _0x34b6[1009];
  html2canvas(this[_0x34b6[2224]], {
    allowTaint : true,
    taintTest : false,
    onrendered : function(local$$78175) {
      local$$78146[_0x34b6[2224]][_0x34b6[428]][_0x34b6[687]] = _0x34b6[624];
      var local$$78194 = local$$78175[_0x34b6[1012]]();
      /** @type {!Image} */
      var local$$78197 = new Image;
      local$$78197[_0x34b6[551]] = local$$78194;
      /**
       * @return {undefined}
       */
      local$$78197[_0x34b6[443]] = function() {
        var local$$78208 = undefined;
        local$$78208 = document[_0x34b6[424]](_0x34b6[516]);
        local$$78208[_0x34b6[208]] = local$$78197[_0x34b6[208]];
        local$$78208[_0x34b6[209]] = local$$78197[_0x34b6[209]];
        if (local$$78146[_0x34b6[1115]] == _0x34b6[381]) {
          /** @type {number} */
          local$$78146[_0x34b6[1688]] = local$$78208[_0x34b6[208]] / local$$78208[_0x34b6[209]];
        }
        var local$$78261 = local$$78208[_0x34b6[403]](_0x34b6[402]);
        local$$78261[_0x34b6[542]](local$$78197, 0, 0, local$$78208[_0x34b6[208]], local$$78208[_0x34b6[209]]);
        if (local$$78146[_0x34b6[268]] == undefined) {
          local$$78146[_0x34b6[268]] = new THREE.SpriteMaterial({
            depthTest : false,
            map : new THREE.CanvasTexture(local$$78208)
          });
          local$$78146[_0x34b6[1684]] = new LSJBillboard(local$$78146[_0x34b6[268]], local$$78146, {
            verticalOrign : local$$78146[_0x34b6[1349]],
            horizontalOrigin : local$$78146[_0x34b6[1350]]
          });
          local$$78146[_0x34b6[1684]][_0x34b6[430]][_0x34b6[338]](local$$78146[_0x34b6[430]]);
          local$$78146[_0x34b6[1446]][_0x34b6[274]](local$$78146[_0x34b6[1684]]);
        } else {
          local$$78146[_0x34b6[268]][_0x34b6[645]][_0x34b6[554]] = local$$78208;
          /** @type {boolean} */
          local$$78146[_0x34b6[268]][_0x34b6[645]][_0x34b6[1275]] = true;
        }
        local$$78146[_0x34b6[243]] = _0x34b6[1710];
        /** @type {boolean} */
        local$$78146[_0x34b6[1685]] = false;
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
LSJGeoLabel[_0x34b6[219]][_0x34b6[1241]] = function(local$$78391, local$$78392, local$$78393) {
  this[_0x34b6[430]][_0x34b6[290]] = local$$78391;
  this[_0x34b6[430]][_0x34b6[291]] = local$$78392;
  this[_0x34b6[430]][_0x34b6[1287]] = local$$78393;
  if (this[_0x34b6[1684]] != undefined) {
    this[_0x34b6[1684]][_0x34b6[430]][_0x34b6[338]](this[_0x34b6[430]]);
  }
};
/**
 * @return {?}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[1240]] = function() {
  return this[_0x34b6[1690]];
};
/**
 * @return {?}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[1288]] = function() {
  return this[_0x34b6[1686]];
};
/**
 * @param {string} local$$78479
 * @param {?} local$$78480
 * @return {undefined}
 */
LSJGeoLabel[_0x34b6[219]][_0x34b6[225]] = function(local$$78479, local$$78480) {
  if (this[_0x34b6[1685]] && local$$78480) {
    this[_0x34b6[1261]]();
  }
  if (this[_0x34b6[1684]] != undefined) {
    this[_0x34b6[1684]][_0x34b6[240]] = getCamera();
    this[_0x34b6[1684]][_0x34b6[1263]]();
    local$$78479[_0x34b6[1692]][_0x34b6[1263]]();
    local$$78479[_0x34b6[1692]][_0x34b6[1620]]();
    var local$$78536 = new THREE.Vector3(0, 0, 0);
    var local$$78543 = new THREE.Vector3(0, 0, 0);
    local$$78543[_0x34b6[338]](this[_0x34b6[1684]][_0x34b6[430]]);
    /** @type {number} */
    var local$$78557 = 1;
    /** @type {number} */
    var local$$78560 = 1;
    var local$$78570 = getCamera()[_0x34b6[430]][_0x34b6[1593]](local$$78543);
    if (local$$78570 > local$$78479[_0x34b6[1447]][_0x34b6[1693]] && local$$78479[_0x34b6[1447]][_0x34b6[1693]] != 0) {
      var local$$78626 = getCamera()[_0x34b6[430]][_0x34b6[212]]()[_0x34b6[274]](getCamera()[_0x34b6[430]][_0x34b6[212]]()[_0x34b6[1434]](local$$78543)[_0x34b6[1487]]()[_0x34b6[350]](local$$78479[_0x34b6[1447]][_0x34b6[1693]]));
      var local$$78639 = local$$78626[_0x34b6[212]]()[_0x34b6[1477]](local$$78479[_0x34b6[1692]]);
      /** @type {number} */
      var local$$78658 = (local$$78639[_0x34b6[290]] + 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[545]];
      var local$$78665 = new THREE.Vector3(0, 1, 0);
      local$$78665[_0x34b6[1696]](local$$78479[_0x34b6[1692]][_0x34b6[1239]]);
      var local$$78693 = local$$78626[_0x34b6[212]]()[_0x34b6[274]](local$$78665)[_0x34b6[1477]](local$$78479[_0x34b6[1692]]);
      /** @type {number} */
      var local$$78713 = -(local$$78639[_0x34b6[291]] - 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      var local$$78733 = -(local$$78693[_0x34b6[291]] - 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      var local$$78742 = 1 / Math[_0x34b6[1525]](local$$78713 - local$$78733);
      local$$78557 = Math[_0x34b6[1525]](local$$78742);
      local$$78639 = local$$78543[_0x34b6[212]]()[_0x34b6[1477]](local$$78479[_0x34b6[1692]]);
      local$$78665 = new THREE.Vector3(0, 1, 0);
      local$$78665[_0x34b6[1696]](local$$78479[_0x34b6[1692]][_0x34b6[1239]]);
      local$$78693 = local$$78543[_0x34b6[212]]()[_0x34b6[274]](local$$78665)[_0x34b6[1477]](local$$78479[_0x34b6[1692]]);
      /** @type {number} */
      local$$78713 = -(local$$78639[_0x34b6[291]] - 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      local$$78733 = -(local$$78693[_0x34b6[291]] - 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      var local$$78845 = 1 / Math[_0x34b6[1525]](local$$78713 - local$$78733);
      if (local$$78845 > 2 * local$$78742) {
        /** @type {number} */
        local$$78557 = local$$78845 / 2;
      }
      /** @type {number} */
      local$$78560 = local$$78742 / local$$78845;
    } else {
      local$$78639 = local$$78543[_0x34b6[212]]()[_0x34b6[1477]](local$$78479[_0x34b6[1692]]);
      local$$78665 = new THREE.Vector3(0, 1, 0);
      local$$78665[_0x34b6[1696]](local$$78479[_0x34b6[1692]][_0x34b6[1239]]);
      local$$78693 = local$$78543[_0x34b6[212]]()[_0x34b6[274]](local$$78665)[_0x34b6[1477]](local$$78479[_0x34b6[1692]]);
      /** @type {number} */
      local$$78713 = -(local$$78639[_0x34b6[291]] - 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      local$$78733 = -(local$$78693[_0x34b6[291]] - 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[548]];
      /** @type {number} */
      local$$78557 = 1 / Math[_0x34b6[1525]](local$$78713 - local$$78733);
      /** @type {number} */
      local$$78658 = (local$$78639[_0x34b6[290]] + 1) / 2 * local$$78479[_0x34b6[1695]][_0x34b6[1694]][_0x34b6[545]];
    }
    if (this[_0x34b6[2224]] != undefined) {
      /** @type {number} */
      this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[290]] = this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[291]] = this[_0x34b6[1684]][_0x34b6[1090]][_0x34b6[1287]] = this[_0x34b6[2223]] * local$$78557;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[432]] = local$$78658;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[656]] = local$$78713;
      /** @type {number} */
      this[_0x34b6[1686]][_0x34b6[434]] = local$$78713 + this[_0x34b6[2223]] * local$$78560;
    }
    /** @type {number} */
    var local$$79064 = this[_0x34b6[1686]][_0x34b6[434]] - this[_0x34b6[1686]][_0x34b6[656]];
    /** @type {number} */
    var local$$79070 = local$$79064 * this[_0x34b6[1688]];
    this[_0x34b6[1686]][_0x34b6[655]] = this[_0x34b6[1686]][_0x34b6[432]] + local$$79070;
    if (local$$78479 != undefined) {
      local$$78479[_0x34b6[1276]][_0x34b6[220]](this[_0x34b6[1684]]);
    }
  }
};
