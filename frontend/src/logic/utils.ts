export function arrayBufferToImageSrcB64(
  image: ArrayBuffer | null,
  imageType: "png" | "jpeg"
): string | null {
  return image === null
    ? image
    : `data:image/${imageType};base64,${btoa(
        new Uint8Array(image).reduce(
          (acc, current) => acc + String.fromCharCode(current),
          ""
        )
      )}`;
}
