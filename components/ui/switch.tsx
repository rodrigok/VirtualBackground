"use client";

import * as SwitchPrimitive from "@radix-ui/react-switch";
import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

export function Switch({ className, ...props }: ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root className={cn("peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border border-transparent bg-white/10 transition-colors data-[state=checked]:bg-cyan-400", className)} {...props}>
      <SwitchPrimitive.Thumb className="pointer-events-none block size-4 translate-x-0.5 rounded-full bg-slate-300 shadow-sm transition-transform data-[state=checked]:translate-x-[17px] data-[state=checked]:bg-slate-950" />
    </SwitchPrimitive.Root>
  );
}
